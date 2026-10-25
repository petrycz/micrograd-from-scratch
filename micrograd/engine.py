"""
engine.py — the autograd core you will build.

You write this file by following GUIDE.md:
  Stage 1  ->  Value.__init__, __add__, __mul__
  Stage 2  ->  backward()
  Stage 3  ->  __pow__, relu, tanh, exp, and the dunder conveniences

Until you implement each piece, the matching tests in tests/test_engine.py
fail on purpose (red -> green). Delete the NotImplementedError lines as you go.
"""
import math


class Value:
    """A single scalar and its gradient — one node in the computation graph."""

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = _children
        self._op = _op
        self._backward = lambda: None
        # Stage 1 — store: data (float), grad (0.0), _backward (a no-op closure),
        # _prev (set of child Values), _op (a label string, for debugging).

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        result = Value(other.data+self.data, {self, other}, '+')
        def addgrad():
            #making the ones explicit - we multiply by chain rule, and if a node was created via n = a+b from a sum of other nodes, the local derivative wr to each of those input nodes is just 1
            self.grad+=1*result.grad
            other.grad+=1*result.grad

        result._backward = addgrad
        return result

    def __mul__(self, other):
        # the local derivative of n = a*b is just the other factor (eg. dn/da = b etc)
        other = other if isinstance(other, Value) else Value(other)
        result = Value(other.data*self.data, {self, other}, '*')
        def mulgrad():
            self.grad+=result.grad*other.data
            other.grad+=result.grad*self.data

        result._backward=mulgrad 
        return result

    def backwards(self):
        self.grad = 1.0
        stack1 = [self]
        stack2 = []
        # postorder traversal
        while stack1:
            node = stack1.pop()
            stack2.append(node)
            stack1.extend(node._prev)
        
        for i in stack2:
            i._backward()
        

    def backward(self):
        seen=set()
        seen.add(self)
        res = []
        def explore(c, res, seen):
            if c._prev is None:
                return

            for child in c._prev:
                if not child in seen:
                    seen.add(child)
                    explore(child, res, seen)

            res.append(c)

        explore(self, res, seen)
        self.grad = 1.0
        for r in reversed(res):
            r._backward()





    # ---- Stage 3: add __pow__, relu, tanh, exp, and the conveniences below ----
    # __neg__, __radd__, __sub__, __rsub__, __rmul__, __truediv__, __rtruediv__, __repr__

    def __pow__(self, k):
        
        result = Value(self.data**k, {self}, '**')
        def powergrad():
            self.grad+=result.grad*k*self.data**(k-1)
        
        result._backward = powergrad
        return result


    def relu(self):
        result = Value(max(0,self.data), {self}, 'rel')
        def relgrad():
            self.grad+=result.grad*(self.data>0)

        result._backward = relgrad
        return result


    def tanh(self):
        t = math.tanh(self.data)
        result = Value(t, {self}, 'tanh')
        def tanhgrad():
            self.grad+=result.grad*(1-t**2)

        result._backward = tanhgrad
        return result


    def exp(self):
        result = Value(math.exp(self.data), {self}, 'expo')
        def expograd():
            self.grad+=result.grad*result.data

        result._backward = expograd
        return result


    def __neg__(self):       return self * -1
    def __radd__(self, o):   return self + o
    def __sub__(self, o):    return self + (-o)
    def __rsub__(self, o):   return o + (-self)
    def __rmul__(self, o):   return self * o
    def __truediv__(self, o):return self * o ** -1
    def __rtruediv__(self,o):return o * self ** -1
    def __repr__(self):      return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
