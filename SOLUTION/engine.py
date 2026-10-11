import math

class Value:
    """A single scalar value and its gradient — one node in the computation graph."""
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None     # closure: push grad to inputs
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += out.grad        # local deriv d(a+b)/da = 1
            other.grad += out.grad        # local deriv d(a+b)/db = 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # d(a*b)/da = b
            other.grad += self.data  * out.grad   # d(a*b)/db = a
        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only int/float powers"
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad   # d(a^k)/da = k a^(k-1)
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad   # 1 if active, else 0
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t * t) * out.grad      # d tanh/dx = 1 - tanh^2
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad         # d e^x/dx = e^x
        out._backward = _backward
        return out

    def backward(self):
        # build reverse-topological order, then sweep, seeding dL/dL = 1
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v._prev:
                    build(c)
                topo.append(v)
        build(self)
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

    def __neg__(self):      return self * -1
    def __radd__(self, o):  return self + o
    def __sub__(self, o):   return self + (-o)
    def __rsub__(self, o):  return o + (-self)
    def __rmul__(self, o):  return self * o
    def __truediv__(self,o):return self * o ** -1
    def __rtruediv__(self,o):return o * self ** -1
    def __repr__(self):     return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
