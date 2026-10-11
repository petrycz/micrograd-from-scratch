"""
nn.py — a tiny neural-net library built on your Value autograd (GUIDE Stage 4).

Build Module, Neuron, Layer, MLP. Each Neuron/Layer/MLP holds Value parameters
and is callable on a list of Values. parameters() returns every trainable Value.
"""
import random
from .engine import Value


class Module:
    def zero_grad(self):
        raise NotImplementedError("Stage 4")

    def parameters(self):
        return []


class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        raise NotImplementedError("Stage 4")

    def __call__(self, x):
        raise NotImplementedError("Stage 4")

    def parameters(self):
        raise NotImplementedError("Stage 4")


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        raise NotImplementedError("Stage 4")

    def __call__(self, x):
        raise NotImplementedError("Stage 4")

    def parameters(self):
        raise NotImplementedError("Stage 4")


class MLP(Module):
    def __init__(self, nin, nouts):
        raise NotImplementedError("Stage 4")

    def __call__(self, x):
        raise NotImplementedError("Stage 4")

    def parameters(self):
        raise NotImplementedError("Stage 4")
