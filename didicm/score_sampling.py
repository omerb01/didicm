import abc
import torch
import torch.nn.functional as F

from .utils import noise_utils, score_utils


# ----------------------------------------------------------------------------------------------------------
#                                       Predictor Classes
# ----------------------------------------------------------------------------------------------------------


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, num_classes, noise_type):
        super().__init__()
        self.num_classes = num_classes
        self.noise = noise_utils.get_noise(noise_type=noise_type)

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


class ConditionalSEDDPredictor(Predictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = score_utils.Uniform(dim=self.num_classes)


class EulerPredictor(ConditionalSEDDPredictor):
    def update_fn(self, model, y, label, t, step_size):
        sigma, dsigma = self.noise(t)
        
        # Get score from model
        score = score_utils.score_fn(model=model, y=y, labels=label, sigma=sigma, sampling=True)

        # Fix: Reshape dsigma to [batch_size, 1] for proper broadcasting with normalized_rate
        dsigma_reshaped = dsigma.view(-1, 1)
        
        # Apply step size and dsigma using the reshaped tensor
        rev_rate = step_size * dsigma_reshaped * self.graph.reverse_rate(label, score)
        
        # Sample the next state
        next_label = self.graph.sample_rate(label, rev_rate)
        
        return next_label


class AnalyticPredictor(ConditionalSEDDPredictor):
    def update_fn(self, model, y, label, t, step_size):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_utils.score_fn(model=model, y=y, labels=label, sigma=curr_sigma, sampling=True)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(label, dsigma)
        return score_utils.sample_categorical(probs)


class DiDiCMPredictor(Predictor):
    def __init__(self, mode, *args, **kwargs,):
        super().__init__(*args, **kwargs)
        self.forward_mat = torch.full((self.num_classes, self.num_classes), 1) - self.num_classes * torch.eye(self.num_classes)
        self.forward_mat /= self.num_classes  # normalization
        self.mode = mode

    def update_fn(self, model, y, p, t, step_size):
        sigma, dsigma = self.noise(t)

        # Get all scores from model
        if self.mode == 'min':
            labels = p.argmin(dim=1)
        elif self.mode == 'max':
            labels = p.argmax(dim=1)
        elif self.mode == 'random':
            labels = score_utils.sample_categorical(p)
        all_scores = score_utils.score_fn(model=model, y=y, labels=labels, sigma=sigma, sampling=True)
        all_scores /= all_scores.sum(dim=1, keepdim=True)
        all_scores = all_scores.unsqueeze(2) * (1 / all_scores.unsqueeze(1))

        # Calculate reverse rates
        all_rev_rates = all_scores * self.forward_mat.unsqueeze(0)
        all_rev_rates -= torch.diag_embed(all_rev_rates.sum(dim=1))

        # Apply step size and dsigma
        dsigma_reshaped = dsigma.view(-1, 1, 1)
        all_rev_rates *= step_size * dsigma_reshaped

        # Add identity matrix to compute transition matrix
        all_rev_rates += torch.eye(self.num_classes, device=p.device)

        # Calculate next probabilities
        next_probs = torch.bmm(all_rev_rates, p.unsqueeze(2)).squeeze()    
        return next_probs


# ----------------------------------------------------------------------------------------------------------
#                                       Sampler Classes
# ----------------------------------------------------------------------------------------------------------


class DiDiCMSampler(abc.ABC):
    """The abstract class for a DiDiC sampler algorithm."""

    def __init__(self, num_classes, noise_type, steps, eps=1e-5, return_diffusion_steps=False, num_steps_to_return=8):
        super().__init__()

        if steps < num_steps_to_return:
            num_steps_to_return = steps

        self.num_classes = num_classes
        self.noise_type = noise_type
        self.steps = steps
        self.eps = eps
        self.return_diffusion_steps = return_diffusion_steps
        self.num_steps_to_return = num_steps_to_return

    @abc.abstractmethod
    @torch.no_grad()
    def run(self, model, y):
        pass


class DiDiCMCPSampler(DiDiCMSampler):

    def __init__(self, mode='min', *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.predictor = DiDiCMPredictor(mode=self.mode, num_classes=self.num_classes, noise_type=self.noise_type)
    
    @torch.no_grad()
    def run(self, model, y):

        all_probs = []
        
        # Initialize probs of the uniform distribution
        probs = torch.full((y.shape[0], self.num_classes), 1 / self.num_classes).to(y.device)
        if self.return_diffusion_steps:
            assert self.steps % self.num_steps_to_return == 0, "steps must be divisible by num_steps_to_return"
            all_probs.append(probs.cpu())

        timesteps = torch.linspace(1, self.eps, self.steps, device=y.device)
        dt = (1 - self.eps) / self.steps

        self.predictor.forward_mat = self.predictor.forward_mat.to(y.device)
        
        for i in range(self.steps):
            t = timesteps[i] * torch.ones(probs.size(0), device=y.device)
            
            # Update probs using the predictor
            probs = self.predictor.update_fn(model, y, probs, t, dt)

            # normalize between 0 and 1 for numerical stability
            probs = probs.clamp(min=0, max=1)
            probs /= probs.sum(dim=-1, keepdim=True)

            if self.return_diffusion_steps and (i+1) % (self.steps // self.num_steps_to_return) == 0:
                all_probs.append(probs.cpu())
        
        if self.return_diffusion_steps:
            return probs, torch.stack(all_probs, dim=1)
        else:
            return probs


class DiDiCMCLSampler(DiDiCMSampler):
    def __init__(self, N=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph = score_utils.Uniform(dim=self.num_classes)
        self.predictor = EulerPredictor(num_classes=self.num_classes, noise_type=self.noise_type)
        self.N = N

    @torch.no_grad()
    def run(self, model, y):

        all_diffusions_labels = []

        for _ in range(self.N):
            all_labels = []

            # Initialize labels from the uniform distribution
            # Following the original implementation, use batch_dims as-is
            sample_uniform_labels = self.graph.sample_limit(y.shape[0]).to(y.device)
            if self.return_diffusion_steps:
                assert self.steps % self.num_steps_to_return == 0, "steps must be divisible by num_steps_to_return"
                all_labels.append(sample_uniform_labels.cpu())

            timesteps = torch.linspace(1, self.eps, self.steps, device=y.device)
            dt = (1 - self.eps) / self.steps
            
            for i in range(self.steps):
                t = timesteps[i] * torch.ones(sample_uniform_labels.size(0), device=y.device)
                
                # Update labels using the predictor
                sample_uniform_labels = self.predictor.update_fn(model, y, sample_uniform_labels, t, dt)

                if self.return_diffusion_steps and (i+1) % (self.steps // self.num_steps_to_return) == 0:
                    all_labels.append(sample_uniform_labels.cpu())
            
            all_labels = torch.stack(all_labels, dim=1)
            all_diffusions_labels.append(all_labels)
        
        all_diffusions_labels = torch.stack(all_diffusions_labels, dim=-1)
        probs = F.one_hot(all_diffusions_labels[:, -1], num_classes=self.num_classes).to(y.dtype).mean(dim=1)
        if self.return_diffusion_steps:
            return probs, all_diffusions_labels
        else:
            return probs


def get_sampler(sampler='cp', *args, **kwargs):
    if sampler == 'cp':
        return DiDiCMCPSampler(*args, **kwargs)
    elif sampler == 'cl':
        return DiDiCMCLSampler(*args, **kwargs)
    else:
        raise ValueError(f"Invalid sampler: {sampler}")
