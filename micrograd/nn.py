"""
nn.py — a tiny neural-net library built on your Value autograd (GUIDE Stage 4).

Build Module, Neuron, Layer, MLP. Each Neuron/Layer/MLP holds Value parameters
and is callable on a list of Values. parameters() returns every trainable Value.
"""
import random
# we have the value object which has data, previous, and backward, which allows backward propagation. I think for this stage we just wanna build the larger blocks. So a neuron is the tanh 
import random
from .engine import Value


class Module:
    def zero_grad(self):
        for p in self.parameters(): 
            p.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):
    def __init__(self, nin, nonlin=True):
        self.nin = nin
        self.nonlin = nonlin
        self.weights = [Value(random.uniform(-1,1))for _ in range(self.nin)]
        self.bias = Value(0.0)

    def __call__(self, x):
        cumul = 0
        for i in range(self.nin):
            cumul+=self.weights[i]*x[i]
        
        inp = cumul + self.bias
        if self.nonlin:
            return inp.tanh()

        return inp

    def parameters(self):
        return self.weights + [self.bias]
        


class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.nin = nin
        self.nouts = nout
        self.neurons = [Neuron(self.nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out =  [n(x) for n in self.neurons]
        return out[0] if len(out)==1 else out

    def parameters(self):
        res = []
        for n in self.neurons:
            res.extend(n.parameters())
        
        return res
    


class MLP(Module):
    def __init__(self, nin, nouts):
        self.nin = nin
        self.nouts = nouts
        self.layers = []
        nouts.insert(0, nin)
        for i in range(len(nouts)-1):
            # each next layer's input is the previous layer's output
            self.layers.append(Layer(nouts[i], nouts[i+1], nonlin=(i!=len(nouts)-1)))

    def __call__(self, x):
        inp = x
        i = 0
        while i<len(self.layers):
            inp = self.layers[i](inp)
            i+=1
        return inp[0] if isinstance(inp, list) else inp

    def parameters(self):
        res = []
        for l in self.layers:
            res.extend(l.parameters())
        
        return res
