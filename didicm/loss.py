import torch
from torch import nn as nn

from .utils import noise_utils, score_utils


class ProbabilityNoiser(nn.Module):

    def __init__(self, num_classes, amp_autocast) -> None:
        super().__init__()
        self.amp_autocast = amp_autocast
        
        uniform_rate_matrix = torch.full((num_classes, num_classes), 1) - num_classes * torch.eye(num_classes)
        uniform_rate_matrix /= num_classes  # normalization

        # decompose uniform_rate_matrix into its eigenvalues and eigenvectors
        # using eigh for symmetric matrices (returns real eigenvalues)
        eigenvalues, eigenvectors = torch.linalg.eigh(uniform_rate_matrix)
        self.register_buffer('eigenvalues', eigenvalues)
        self.register_buffer('eigenvectors', eigenvectors)

        # validate that the uniform_rate_matrix is composed by the eigenvectors and eigenvalues
        reconstructed_rate_matrix = self.eigenvectors @ torch.diag(self.eigenvalues) @ self.eigenvectors.T
        max_diff = torch.abs(uniform_rate_matrix - reconstructed_rate_matrix).max().item()
        assert torch.allclose(uniform_rate_matrix, reconstructed_rate_matrix, rtol=1e-3, atol=1e-5), \
            f"Eigendecomposition reconstruction failed. Max difference: {max_diff}"

    # def setup(self, device):
    #     if not self.enable:
    #         return
        
    #     # cast to float64 for numerical stability
    #     self.eigenvalues = self.eigenvalues.to(device, dtype=torch.float64)
    #     self.eigenvectors = self.eigenvectors.to(device, dtype=torch.float64)

    def forward(self, p_0, sigma):
        
        # Disable amp for numerical stability
        with self.amp_autocast(enabled=False):
            eigenvectors = self.eigenvectors.repeat(p_0.shape[0], 1, 1)
            diags = torch.diag_embed(torch.exp(self.eigenvalues.unsqueeze(0) * sigma.unsqueeze(1)))
            p_t = torch.bmm(eigenvectors, torch.bmm(diags, eigenvectors.transpose(1, 2))).to(dtype=p_0.dtype)
            p_t = torch.bmm(p_t, p_0.unsqueeze(2))[:, :, 0]

        return p_t


# ----------------------------------------------------------------------------------------------------------
#                                       Loss Functions
# ----------------------------------------------------------------------------------------------------------


class DiffusionLoss(nn.Module):

    def __init__(self, noise_type="loglinear", sampling_eps=1e-3):
        super().__init__()
        self.noise = noise_utils.get_noise(noise_type=noise_type)
        self.sampling_eps = sampling_eps

    def forward(self, model, y, p_0, t=None, p_t=None):
        raise NotImplementedError()


class ConditionalSEDDLoss(DiffusionLoss):

    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = score_utils.Uniform(dim=num_classes)

    def forward(self, model, y, c_0, t=None, c_t=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            t = (1 - self.sampling_eps) * torch.rand(c_0.shape[0], device=c_0.device) + self.sampling_eps

        sigma, dsigma = self.noise(t)

        if c_t is None:
            c_t = self.graph.sample_transition(c_0, sigma)

        log_score = score_utils.score_fn(model=model, y=y, labels=c_t, sigma=sigma, sampling=False)
        loss = self.graph.score_entropy(log_score, sigma, c_t, c_0)
        
        # Weight the loss by noise level derivative
        loss = (dsigma * loss).mean(dim=-1)

        return loss


class DiDiCMLoss(DiffusionLoss):

    def __init__(self, num_classes, amp_autocast=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prob_noiser = ProbabilityNoiser(num_classes, amp_autocast)
        self.prob_noiser.to(dtype=torch.float64)  # cast to float64 for numerical stability

    def forward(self, model, y, p_0, t=None, p_t=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            t = (1 - self.sampling_eps) * torch.rand(p_0.shape[0], device=p_0.device) + self.sampling_eps
        
        sigma, dsigma = self.noise(t)

        if p_t is None:
            p_t = self.prob_noiser(p_0, sigma)
        
        # Ensure p_t is properly normalized and has no zeros
        p_t = torch.clamp(p_t, min=1e-10)
        p_t = p_t / p_t.sum(dim=-1, keepdim=True)
        
        c_t = score_utils.sample_categorical(p_t)

        # Calculate the ratios with safety check
        p_t_selected = p_t[range(p_t.shape[0]), c_t].unsqueeze(-1)
        # Clamp to avoid division by very small numbers
        p_t_selected = torch.clamp(p_t_selected, min=1e-10)
        ratios = p_t / p_t_selected
        
        # Ensure ratios are positive and bounded
        ratios = torch.clamp(ratios, min=1e-10, max=1e10)

        # Predict the log score
        log_score = score_utils.score_fn(model=model, y=y, labels=c_t, sigma=sigma, sampling=False)
        
        # Clip log_score to prevent exponential overflow
        log_score = torch.clamp(log_score, min=-20, max=20)

        # Create mask for one-hot encoding
        kron_delta = torch.zeros_like(log_score).scatter_(-1, c_t.unsqueeze(-1).long(), 1.0)

        # Calculate the loss
        K = lambda a: a * (torch.log(torch.clamp(a, min=1e-10)) - 1)
        loss = ((1 - kron_delta) * (log_score.exp() - ratios * log_score) + K(ratios)).mean(dim=-1)
        
        # Weight the loss by noise level derivative
        loss = (dsigma * loss).mean(dim=-1)

        return loss


def get_loss_fn(num_classes, noise_type="loglinear", sampling_eps=1e-3, amp_autocast=None, mixup_active=False):
    if mixup_active:
        return DiDiCMLoss(num_classes, noise_type=noise_type, sampling_eps=sampling_eps, amp_autocast=amp_autocast)
    else:
        return ConditionalSEDDLoss(num_classes, noise_type=noise_type, sampling_eps=sampling_eps)
