from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch.distributions import Beta, Normal, Uniform, Laplace, Exponential, Gamma


def get_distribution(name, **kwargs):
    name = name.lower()
    if name == 'normal':
        distribution = Normal(kwargs['mu'], kwargs['std'])
    elif name == 'uniform':
        distribution = Uniform(kwargs['low'], kwargs['high'])
    elif name == 'beta':
        distribution = Beta(kwargs['a'], kwargs['b'])
    elif name == 'laplace':
        distribution = Laplace(kwargs['a'], kwargs['b'])
    else:
        assert False

    return distribution


def get_trainable_mask(size, initial_distribution):
    if initial_distribution['name'].lower() == 'constant':
        t = torch.empty(size).fill_(initial_distribution['c'])
    else:
        t = get_distribution(**initial_distribution).sample(size)

    if initial_distribution.get('trainable', True):
        return nn.Parameter(t, requires_grad=True)
    return t


class TrainableMask(nn.Module, ABC):
    def __init__(self, t=1):
        super().__init__()
        self.t = t

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calculate_divergence(self, prior: torch.distributions.Distribution):
        raise NotImplementedError


class TrainableBeta(TrainableMask):
    def __init__(self, dimensions, initialization, eps=1e-6, t=1):
        super().__init__(t)
        self.a = get_trainable_mask(dimensions, initialization['a'])
        self.b = get_trainable_mask(dimensions, initialization['b'])
        self.eps = eps

    @property
    def posterior(self) -> torch.distributions.Distribution:
        a, b = torch.relu(self.a) + self.eps, torch.relu(self.b) + self.eps
        return Beta(a, b)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return f'{self.__class__.__name__}(a={tuple(self.a.shape)}, b={tuple(self.b.shape)})'


class TrainableGamma(TrainableMask):
    def __init__(self, dimensions, initialization, eps=1e-6, t=1):
        super().__init__(t)
        self.a = get_trainable_mask(dimensions, initialization['a'])
        self.b = get_trainable_mask(dimensions, initialization['b'])
        self.eps = eps

    @property
    def posterior(self) -> torch.distributions.Distribution:
        a, b = torch.relu(self.a) + self.eps, torch.relu(self.b) + self.eps
        return Gamma(a, b)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return f'{self.__class__.__name__}(a={tuple(self.a.shape)}, b={tuple(self.b.shape)})'


class TrainableLaplace(TrainableMask):
    def __init__(self, dimensions, initialization, t=1):
        super().__init__(t)
        self.mu = get_trainable_mask(dimensions, initialization['mu'])
        self.b = get_trainable_mask(dimensions, initialization['b'])

    @property
    def posterior(self) -> torch.distributions.Distribution:
        return Laplace(self.mu, self.b)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return f'{self.__class__.__name__}(a={tuple(self.mu.shape)}, b={tuple(self.b.shape)})'


class TrainableNormal(TrainableMask):
    def __init__(self, dimensions, initialization, t=1):
        super().__init__(t)
        self.mu = get_trainable_mask(dimensions, initialization['mu'])
        self.std = get_trainable_mask(dimensions, initialization['std'])

    @property
    def posterior(self) -> torch.distributions.Distribution:
        return Normal(self.mu, self.std)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return f'{self.__class__.__name__}(a={tuple(self.mu.shape)}, b={tuple(self.b.shape)})'


class TrainableWeights(TrainableMask):
    def __init__(self, dimensions, initialization):
        super().__init__()
        self.w = get_trainable_mask(dimensions, initialization)

    def __call__(self, *args, **kwargs):
        return self.w

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        # kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return 0

    def __repr__(self):
        return f'{self.__class__.__name__}(dim={tuple(self.w.shape)})'


class TrainableExponential(TrainableMask):
    def __init__(self, dimensions, initialization, eps=1e-12, max=None, t=1):
        super().__init__(t)
        self.l = get_trainable_mask(dimensions, initialization)
        self.eps = eps
        self.max = max

    @property
    def posterior(self) -> torch.distributions.Distribution:
        l = torch.clamp(self.l, min=self.eps, max=self.max)
        return Exponential(l)

    def __call__(self, size=None, reduce=True, *args, **kwargs):
        if size is None:
            size = self.t
        if isinstance(size, int):
            size = torch.Size([size])
        sampled = self.posterior.rsample(size)
        if reduce:
            sampled = sampled.mean(0)
        return sampled

    def calculate_divergence(self, prior: torch.distributions.Distribution):
        kl = torch.distributions.kl.kl_divergence(self.posterior, prior).sum()
        return kl

    def __repr__(self):
        return f'{self.__class__.__name__}(l={tuple(self.l.shape)})'
