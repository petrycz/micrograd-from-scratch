# Build Your Own Autograd Engine — a guided capstone

**What you'll build:** the core of a deep-learning framework — reverse-mode
automatic differentiation — and then use it to train a neural network that learns a
curved decision boundary. By the end you'll have a working `Value` type whose
`.backward()` computes every gradient in an arbitrary expression, a small `nn`
library on top of it, and a training run that hits 100% on two-moons.

**Why this is the right first capstone:** it is the matrix-calculus module made
executable. Everything you need you already derived there:

- the **chain rule** — the gradient of a composition is a product of local derivatives;
- **backpropagation** — seed `∂L/∂L = 1` at the scalar output and sweep *right-to-left*,
  each node computing *(gradient arriving from its right) × (its own local derivative)*;
- the **local derivatives** themselves — the identity table: `+` passes gradient through,
  `*` swaps the operands in, `xⁿ → n·xⁿ⁻¹`, and activations contribute their 1-D slope.

Keep the backprop beat's one-neuron picture in mind — this project is that exact
sweep, scaled up to thousands of nodes.

**How to work:** each stage has a short **concept**, a **your task** (try it before
reading the code), the **code** with commentary, and a **checkpoint** (a test to run).
The suite starts fully red; you'll turn it green group by group. `SOLUTION/` holds a
reference if you get stuck — but type your own version first; the point is the doing.

---

## Stage 0 — Setup (on your machine)

```bash
# in an empty folder (this one):
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # everything fails / errors — expected
```

You should see import or `NotImplementedError` failures. That's your to-do list.
Open `micrograd/engine.py`; you'll fill it in next.

---

## Stage 1 — The `Value` node, and `+` and `*`

**Concept.** A framework can only differentiate a computation it can *see*. So every
number becomes a small object that records three things: its value, a slot for its
gradient (filled in later, during the backward pass), and **who made it** — the inputs
it came from and by what operation. Building an expression therefore builds a graph.

**Your task.** Write `__init__`, then `__add__` and `__mul__`. Each operation returns a
*new* `Value` whose `_prev` is the set of inputs, and attaches a `_backward` closure
that — when later called — pushes gradient from the output back onto those inputs using
the **local derivative**. From the identities: `∂(a+b)/∂a = ∂(a+b)/∂b = 1`, and
`∂(a·b)/∂a = b`, `∂(a·b)/∂b = a`.

> Hint: use `+=` when writing into `self.grad`, never `=`. If a node feeds two places,
> its gradients must **accumulate** (that's the multivariable chain rule summing routes).

**Code.**

```python
import math

class Value:
    """A single scalar and its gradient — one node in the computation graph."""
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None     # no-op until an op sets it
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad  += out.grad        # local derivative 1
            other.grad += out.grad        # local derivative 1
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad  += other.data * out.grad   # ∂(a·b)/∂a = b
            other.grad += self.data  * out.grad   # ∂(a·b)/∂b = a
        out._backward = _backward
        return out
```

**Checkpoint.**

```bash
pytest -q tests/test_engine.py::test_add_mul_forward
```

---

## Stage 2 — `backward()`: the reverse sweep

**Concept.** This *is* backpropagation. Two facts make it work:

1. A node's `_backward` may only run **after** its own `.grad` is final — i.e. after
   every node downstream of it has already run. So we need the nodes in an order where
   each appears after everything that depends on it: a **reverse topological order**.
2. Seed the very end with `∂L/∂L = 1`, then call each node's `_backward` in that reverse
   order. Each call does one thing — *(incoming gradient) × (local derivative)* — exactly
   the per-node rule from the beat.

**Your task.** Build the topological order with a depth-first traversal, set the output's
`grad = 1.0`, and run the closures in reverse.

**Code.**

```python
    def backward(self):
        topo, visited = [], set()
        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)        # appended after its children -> reverse-topo
        build(self)
        self.grad = 1.0               # seed ∂L/∂L = 1
        for v in reversed(topo):
            v._backward()             # upstream × local, node by node
```

**Checkpoint.** This test is the backprop beat's exact one-neuron example — same
numbers you stepped through (`∂L/∂w = −12`-style arithmetic, here with its own values):

```bash
pytest -q tests/test_engine.py::test_backward_single_neuron
pytest -q tests/test_engine.py::test_accumulation_reused_node
```

---

## Stage 3 — More operations, and gradient checking

**Concept.** Each new op is just a (forward value, local-derivative) pair — the identity
table again. Powers give `x**k -> k·x**(k-1)`; `relu` passes gradient only where it's
active (a 0/1 gate — the diagonal activation Jacobian); `tanh` contributes `1 − tanh²`;
`exp` contributes itself. Subtraction, division and negation are *defined in terms of*
`+`, `*` and `**`, so they need no new backward code.

**Your task.** Add `__pow__`, `relu`, `tanh`, `exp`, and the dunder conveniences. Then
run the **gradient check**: compare your analytic gradients to a finite-difference
estimate `(f(x+ε) − f(x−ε)) / 2ε`. If they match to ~1e-4, your chain rule is correct.

**Code.**

```python
    def __pow__(self, k):
        assert isinstance(k, (int, float))
        out = Value(self.data ** k, (self,), f'**{k}')
        def _backward():
            self.grad += (k * self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,), 'relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t * t) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        out = Value(math.exp(self.data), (self,), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):       return self * -1
    def __radd__(self, o):   return self + o
    def __sub__(self, o):    return self + (-o)
    def __rsub__(self, o):   return o + (-self)
    def __rmul__(self, o):   return self * o
    def __truediv__(self, o):return self * o ** -1
    def __rtruediv__(self,o):return o * self ** -1
    def __repr__(self):      return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

**Checkpoint.**

```bash
pytest -q tests/test_engine.py        # all engine tests, incl. the gradient check
```

---

## Stage 4 — A neural net on top: `Neuron`, `Layer`, `MLP`

**Concept.** A neuron is `tanh(w·x + b)` — a dot product (sums of `Value*Value`) through
an activation, all built from ops you already have, so it is automatically
differentiable. A layer is a list of neurons; an MLP is a list of layers. The only new
idea is **parameters**: every weight and bias is a `Value` you'll nudge during training,
so each module exposes `parameters()`.

**Your task.** Implement the four classes in `micrograd/nn.py`. The last layer is linear
(`nonlin=False`) so the output isn't squashed.

**Code.**

```python
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
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(0.0)
        self.nonlin = nonlin
    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh() if self.nonlin else act
    def parameters(self):
        return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out
    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP(Module):
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=(i != len(nouts) - 1))
                       for i in range(len(nouts))]
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

**Checkpoint.**

```bash
pytest -q tests/test_nn.py::test_mlp_shapes_and_params
```

---

## Stage 5 — Train it

**Concept.** Training is the optimization loop from the course, now powered by your
autograd: compute a scalar **loss**, call `loss.backward()` to get every parameter's
gradient in one sweep, then step each parameter *against* its gradient (gradient descent).
We use the SVM max-margin (hinge) loss with labels in `{−1, +1}`, plus a little L2
regularization.

**Your task.** Write `train.py`'s loop. Remember to `model.zero_grad()` each step —
gradients accumulate (Stage 1), so you must reset them before each backward pass.

**Code (core of `train.py`).**

```python
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from data import make_moons

def forward_loss(model, X, y, alpha=1e-4):
    preds = [model([Value(xi[0]), Value(xi[1])]) for xi in X]
    losses = [(1 + -yi * pi).relu() for yi, pi in zip(y, preds)]   # hinge
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    acc = sum((yi > 0) == (pi.data > 0) for yi, pi in zip(y, preds)) / len(y)
    return data_loss + reg, acc

def main():
    X, y01 = make_moons(n=100, noise=0.10, seed=1)
    y = y01 * 2 - 1
    model = MLP(2, [16, 16, 1])
    for k in range(100):
        total, acc = forward_loss(model, X, y)
        model.zero_grad()
        total.backward()
        lr = 1.0 - 0.9 * k / 100          # simple learning-rate decay
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0:
            print(f"step {k:3d}   loss {total.data:.4f}   acc {acc*100:.1f}%")
```

**Checkpoint.**

```bash
pytest -q tests/test_nn.py::test_mlp_learns_moons   # trains a small net, asserts >=90%
python train.py                                     # full run -> ~100%
```

> Pure-Python scalar autograd is **slow** (every number is an object; one step builds
> tens of thousands of them). That slowness is the whole motivation for the *next*
> capstone — a vectorized engine on numpy arrays, where a layer is one matrix multiply.

---

## Stage 6 — See the decision boundary

Add the plotting block to `main()` (full version in `SOLUTION/train.py`): sweep a grid
over the plane, classify each point with the trained model, and shade the two regions.
Running `python train.py` writes `boundary.png`. Compare it to
`docs/target_boundary.png` — two cleanly separated moons.

```python
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = 0.25
    xx, yy = np.meshgrid(np.arange(X[:,0].min()-1, X[:,0].max()+1, h),
                         np.arange(X[:,1].min()-1, X[:,1].max()+1, h))
    pts = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model([Value(p[0]), Value(p[1])]).data > 0 for p in pts]).reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3)
    plt.scatter(X[:,0], X[:,1], c=y01, cmap="RdBu", edgecolors="k", s=28)
    plt.savefig("boundary.png", dpi=110)
```

---

## Stage 7 — Ship it to GitHub

```bash
git init
git add .
git commit -m "Build micrograd from scratch: scalar autograd + MLP trained on two-moons"

# create an empty repo on github.com (no README), then:
git remote add origin git@github.com:<you>/micrograd-from-scratch.git
git branch -M main
git push -u origin main
```

`.gitignore` already excludes `.venv/`, caches and the generated `boundary.png`. Decide
whether to keep `SOLUTION/` in your history (you can `git rm -r SOLUTION` first if you'd
rather the repo be only your own work).

---

## Stretch goals (your own work — the real learning)

1. **Add an op with a gradient check.** Implement `sigmoid` (or `log`); add a test that
   finite-difference-checks its gradient inside a random expression.
2. **Swap the loss.** Replace hinge with a softmax + cross-entropy for a 2-class output;
   confirm it still separates the moons.
3. **Visualize the graph.** Use `graphviz` to draw a small expression's DAG with each
   node's data and grad — you'll *see* the reverse sweep.
4. **Sanity-check against PyTorch.** Rebuild one expression with `torch` tensors
   (`requires_grad=True`), call `.backward()`, and confirm the gradients match yours.
5. **Feel the cost.** Time a training step as you grow the net; plot step-time vs
   parameter count. This is the empirical case for the vectorized engine next.

## How the code maps to the math

| operation (code) | local derivative | from the module |
|---|---|---|
| `a + b` | `1`, `1` | sum node passes gradient through |
| `a * b` | `b`, `a` | product rule entries |
| `a ** k` | `k·a**(k-1)` | power rule |
| `relu(x)` | `1` if `x>0` else `0` | diagonal activation Jacobian |
| `tanh(x)` | `1 − tanh²x` | elementwise map → diagonal |
| `backward()` | product of locals, reversed | chain rule = product of Jacobians, swept right-to-left from the scalar loss |
