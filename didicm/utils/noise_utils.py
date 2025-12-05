import os
from argparse import Namespace
import abc
import math

import torch
import torch.nn.functional as F
from torch import nn as nn

# ----------------------------------------------------------------------------------------------------------
#                                   Noise scheduling classes
# ----------------------------------------------------------------------------------------------------------
class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """

    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass


class GeometricNoise(Noise):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[1].log() - self.sigmas[0].log())

    def total_noise(self, t):
        return self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t


class LogLinearNoise(Noise):
    """
    -- Taken from the SEDD github
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing
    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)


def get_noise(noise_type="loglinear", sigma_min=1e-3, sigma_max=1):
    """
    Helper function to create noise schedulers.
    -- I used loglinear

    Args:
        noise_type: Type of noise ("geometric" or "loglinear")
        sigma_min: Minimum sigma value for geometric noise
        sigma_max: Maximum sigma value for geometric noise

    Returns:
        Noise scheduler
    """
    if noise_type == "geometric":
        return GeometricNoise(sigma_min, sigma_max)
    elif noise_type == "loglinear":
        return LogLinearNoise()
    else:
        raise ValueError(f"{noise_type} is not a valid noise type")