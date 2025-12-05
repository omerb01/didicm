import abc
import torch
import torch.nn.functional as F
from torch import nn as nn


def sample_categorical(categorical_probs, method="hard"):
    """
    Sample from a categorical distribution.

    Args:
        categorical_probs: Probability vectors [batch_size, num_classes]
        method: Sampling method ("hard" for Gumbel-Max sampling)

    Returns:
        Sampled indices [batch_size]
    """
    if method == "hard":
        gumbel_norm = 1e-10 - (torch.rand_like(categorical_probs) + 1e-10).log()
        return (categorical_probs / gumbel_norm).argmax(dim=-1)
    else:
        raise ValueError(f"Method {method} for sampling categorical variables is not valid.")


# ----------------------------------------------------------------------------------------------------------
#                                   The Concrete Score Function
# ----------------------------------------------------------------------------------------------------------


def score_fn(model, y, labels, sigma, sampling=False):
    """
    Compute scores for input data and labels at time t.

    Args:
        y: Input data - degraded image [batch_size, channels, height, width]
        labels: Labels [batch_size]
        sigma: Time values [batch_size]

    Returns:
        Score values
    """
    sigma = sigma.reshape(-1)
    score = model(y, c=labels, t=sigma)

    # Create mask for one-hot encoding
    mask = torch.zeros_like(score).scatter_(-1, labels.unsqueeze(-1).long(), 1.0)
    score = score * (1 - mask)

    if sampling:
        # For sampling, return exp(score) as true score - note from original git
        return score.exp()
    return score


# ----------------------------------------------------------------------------------------------------------
#                                   Uniform - Transition matrix ( As proposed by SEDD )
# ----------------------------------------------------------------------------------------------------------

class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass

    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass

    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass

    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass

    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        # Compute the transpose rate
        normalized_rate = self.transp_rate(i) * score
        
        # Zero out diagonal and normalize
        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate[..., :1]))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        
        return normalized_rate

    def sample_rate(self, i, rate):
        # Create one-hot encoding for current labels
        one_hot = F.one_hot(i, num_classes=self.dim).to(rate.device)
        
        return sample_categorical(one_hot.to(rate) + rate)

    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass

    @abc.abstractmethod
    def score_entropy(self, score, sigma, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """

    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        # The transpose rate is the same as the rate for uniform
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transition(self, i, sigma):
        # Reshape sigma for broadcasting
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1)
        
        # Calculate transition probability
        exp_term = (-sigma).exp()
        prob_change = 1 - exp_term
        
        # Create transition matrix
        trans = torch.ones(*i.shape, self.dim, device=i.device) * prob_change / self.dim
        
        # Set diagonal (staying in same state)
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans[..., :1]))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        
        return trans

    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        # Make sure dsigma is properly shaped for broadcasting
        if dsigma.dim() == 1:
            dsigma = dsigma.view(-1, 1)
        
        dim = score.shape[-1]
        epow = (-dsigma).exp()
        
        # Ensure proper broadcasting by keeping dimensions consistent
        # Sum across the class dimension and keep the batch dimension
        score_sum = score.sum(dim=-1, keepdim=True)
        
        # Calculate the staggered score with proper broadcasting
        result = ((epow - 1) / (dim * epow)) * score_sum + score / epow
        
        return result

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0):
        esigm1 = torch.where(
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim
        )

        # positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const