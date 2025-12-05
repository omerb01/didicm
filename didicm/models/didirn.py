import math
from abc import abstractmethod

import torch
from torch import nn as nn

from timm.models import register_model


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, cond_channels, num_classes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, out_channels),
        )

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels * self.expansion),
            nn.SiLU()
        )

    def forward(self, x, emb, labels, sigma):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        emb_out = self.emb_layers(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]

        out = out + emb_out
        out = self.out_layers(out)
        out += self.skip_connection(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, cond_channels, num_classes, stride=1):
        super(Bottleneck, self).__init__()
        # Bottleneck design: 1x1 -> 3x3 -> 1x1 convolutions
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.skip_connection = nn.Identity()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.skip_connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_channels, out_channels * self.expansion),
        )

        self.out_layers = nn.Sequential(
            GroupNorm32(32, out_channels * self.expansion),
            nn.SiLU()
        )

    def forward(self, x, emb, labels, sigma):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        emb_out = self.emb_layers(emb).type(out.dtype)
        while len(emb_out.shape) < len(out.shape):
            emb_out = emb_out[..., None]

        out = out + emb_out
        out = self.out_layers(out)
        out += self.skip_connection(x)
        out = self.relu(out)
        return out


# Architecture configurations
RESNET_CONFIGS = {
    'resnet18': {'block': BasicBlock, 'layers': [2, 2, 2, 2]},
    'resnet34': {'block': BasicBlock, 'layers': [3, 4, 6, 3]},
    'resnet50': {'block': Bottleneck, 'layers': [3, 4, 6, 3]},
    'resnet101': {'block': Bottleneck, 'layers': [3, 4, 23, 3]},
    'resnet152': {'block': Bottleneck, 'layers': [3, 8, 36, 3]},
}


class DiDiRN(torch.nn.Module):
    def __init__(self,
                 num_classes=1000,
                 arch='resnet18',
                 cond_channels=128,
                 *args, **kwargs):
        super(DiDiRN, self).__init__()

        if arch not in RESNET_CONFIGS:
            raise ValueError(f"Architecture '{arch}' not supported. Choose from: {list(RESNET_CONFIGS.keys())}")

        config = RESNET_CONFIGS[arch]
        block = config['block']
        layers = config['layers']

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.num_classes = num_classes

        self.ff_emb = TimestepEmbedder(cond_channels)
        self.label_embedder = nn.Embedding(num_classes, cond_channels)

        # Build layers dynamically based on architecture
        self.layer1 = self._make_layer(block, 64, layers[0], cond_channels=cond_channels,
                                       num_classes=num_classes, stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], cond_channels=cond_channels,
                                       num_classes=num_classes, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], cond_channels=cond_channels,
                                       num_classes=num_classes, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], cond_channels=cond_channels,
                                       num_classes=num_classes, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # Final FC layer accounts for block expansion
        final_channels = 512 * block.expansion
        self.fc = nn.Linear(final_channels, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, cond_channels, num_classes, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = nn.ModuleList()
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, cond_channels, num_classes, stride))
            self.in_channels = out_channels * block.expansion
        return layers

    def forward(self, x, c, t):
        labels = c
        sigma = t

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        time_emb = self.ff_emb(sigma)
        label_emb = self.label_embedder(labels)
        emb = time_emb + label_emb

        for blk in self.layer1:
            out = blk(out, emb, labels, sigma)
        for blk in self.layer2:
            out = blk(out, emb, labels, sigma)
        for blk in self.layer3:
            out = blk(out, emb, labels, sigma)
        for blk in self.layer4:
            out = blk(out, emb, labels, sigma)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


# Convenience factory functions
@register_model
def didirn18(pretrained=False, **kwargs):
    return DiDiRN(arch='resnet18', **kwargs)


@register_model
def didirn34(pretrained=False, **kwargs):
    return DiDiRN(arch='resnet34', **kwargs)


@register_model
def didirn50(pretrained=False, **kwargs):
    return DiDiRN(arch='resnet50', **kwargs)


@register_model
def didirn101(pretrained=False, **kwargs):
    return DiDiRN(arch='resnet101', **kwargs)


@register_model
def didirn152(pretrained=False, **kwargs):
    return DiDiRN(arch='resnet152', **kwargs)