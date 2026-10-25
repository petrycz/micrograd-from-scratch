# Build Your Own Autograd Engine — a guided capstone

**What you'll build, in one sentence:** a tiny clone of the machinery inside PyTorch —
reverse-mode automatic differentiation — and then use it to train a small neural network
that separates two interleaving crescents of points ("two-moons").

By the end you'll have:
- a `Value` type whose `.backward()` fills in the gradient of *every* number that fed into
  a result — your own autograd engine;
- a small `nn` library (neurons, layers, an MLP) built on top of it;
- a training script that drives accuracy to ~100% and draws the learned boundary.

**What you need to know going in:** only the calculus course you finished. The *engine*
(Stages 1–4) is that course made executable — the chain rule, backpropagation, and the
local-derivative table, nothing more. Stage 5 then *uses* the engine to train a
classifier, which brings in a few ideas from machine learning (what a loss is, why we do
gradient descent, what the hinge loss and L2 are). **Every one of those is built from
pieces you already have, and spelled out here where it comes up** — so this project needs
nothing beyond its own files and the calculus course. No prior ML assumed.

**How the stages are graded (read this once — it removes a lot of confusion):**
- `micrograd/engine.py` and `micrograd/nn.py` are **stubs you fill in**. They're checked by
  `pytest` — the suite starts all-red and you turn it green stage by stage.
- `train.py` is **also a stub, but no test runs it.** It's the capstone *demo*: you write
  it in Stages 5–6, then run `python train.py` yourself and watch it work. The tests check
  your engine and your `nn` library; `train.py` is how *you* see the whole thing learn.
- `data.py` and everything in `tests/` are **already written** — you never edit them.

**How to work each stage:** read the **concept**, attempt the **your task**, then expand the
collapsible **reference implementation** to check yourself. Run the **checkpoint** test. Type
your own version before peeking — the point is the doing. `SOLUTION/` holds the full
reference if you're stuck.

---

## Stage 0 — Setup (on your machine)

```bash
# in an empty folder (this one):
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                            # everything fails / errors — expected
```

You should see import or `NotImplementedError` failures. That's your to-do list. Open
`micrograd/engine.py`; you'll fill it in next.

---

## Stage 1 — The `Value` node, and `+` and `*`

**Concept.** A framework can only differentiate a computation it can *see*. So we wrap every
number in a small object that records three things: its value, a slot for its gradient
(filled in later, during the backward pass), and **who made it** — the inputs it came from
and by what operation. Because each result remembers its inputs, building an expression
automatically builds a **graph** of who-fed-whom. That graph is what we'll walk backward.

Each operation does two jobs. Forward: compute the result. Backward: know how to push
gradient from the result onto its inputs, using the **local derivative** — the same table
from the course: `∂(a+b)/∂a = ∂(a+b)/∂b = 1`, and `∂(a·b)/∂a = b`, `∂(a·b)/∂b = a`.

**Your task.** Write `__init__`, then `__add__` and `__mul__`. Each returns a *new* `Value`
whose `_prev` is its inputs, and attaches a `_backward` closure that — when called later —
adds each input's share of the gradient.

> Hint: write into `self.grad` with `+=`, never `=`. If a node feeds two places downstream,
> its gradient is the **sum** of what comes back from both — that's the multivariable chain
> rule adding up routes. `=` would clobber one of them.

<details>
<summary>Reference implementation — try the task first, then expand</summary>

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

</details>

**Checkpoint.**

```bash
pytest -q tests/test_engine.py::test_add_mul_forward
```

---

## Stage 2 — `backward()`: the reverse sweep

**Concept.** This *is* backpropagation, and it rests on two facts:

1. A node's `_backward` may only run **after** its own `.grad` is final — i.e. after every
   node that depends on it has already contributed. So we need the nodes in an order where
   each comes after everything downstream of it: a **reverse topological order**.
2. Seed the very end with `∂L/∂L = 1` (the output's gradient with respect to itself is 1),
   then call each node's `_backward` in that reverse order. Each call does the one thing from
   the backprop beat — *(gradient arriving from the right) × (local derivative)* — and
   because of `+=`, gradients accumulate correctly where paths rejoin.

**Your task.** Build the order with a depth-first traversal, set the output's `grad = 1.0`,
then run the closures in reverse.

<details>
<summary>Reference implementation — try the task first, then expand</summary>

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

</details>

**Checkpoint.** The first test is the backprop beat's exact one-neuron example — the same
shape of arithmetic you stepped through by hand (`∂L/∂w = −12`-style):

```bash
pytest -q tests/test_engine.py::test_backward_single_neuron
pytest -q tests/test_engine.py::test_accumulation_reused_node
```

---

## Stage 3 — More operations, and gradient checking

**Concept.** Every new op is just a (forward value, local-derivative) pair — the identity
table again. Powers: `x**k → k·x**(k-1)`. `relu`: passes gradient only where it's active (a
0/1 gate — the diagonal activation Jacobian). `tanh`: contributes `1 − tanh²`. `exp`:
contributes itself. Subtraction, division and negation are *defined in terms of* `+`, `*`
and `**`, so they need no new backward code — the chain rule flows through the pieces.

**Your task.** Add `__pow__`, `relu`, `tanh`, `exp`, and the dunder conveniences. Then run a
**gradient check** — the single best proof your engine is correct. Pick a random
expression, compute each input's gradient two ways: (a) your `backward()`, and (b) a
finite-difference estimate `(f(x+ε) − f(x−ε)) / 2ε` that nudges the input and watches the
output. If they agree to ~1e-4, your chain rule is right. (This is independent of any
neural-net code — it tests the math directly.)

<details>
<summary>Reference implementation — try the task first, then expand</summary>

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

</details>

**Checkpoint.**

```bash
pytest -q tests/test_engine.py        # all engine tests, incl. the gradient check
```

---

## Stage 4 — A neural net on top: `Module`, `Neuron`, `Layer`, `MLP`

**Concept.** Everything here is built from the ops you just wrote, so it's all automatically
differentiable. Four classes:

- **`Neuron`** — holds a weight `Value` per input plus one bias `Value`, and computes a
  weighted sum `w·x + b`, then (usually) an activation. In this project a neuron is
  `tanh(w·x + b)`.
- **`Layer`** — a list of neurons all seeing the same input; its output is the list of their
  outputs.
- **`MLP`** — a list of layers; it feeds the input through each in turn.
- **`Module`** — a tiny shared base the other three inherit from. It exists only to provide
  two helpers every piece needs:
  - `parameters()` — collect *all* of this piece's weight/bias `Value`s into one flat list,
    so the training loop can walk them and update each.
  - `zero_grad()` — set every `.grad` back to `0` before each backward pass. This is
    **required** because your Stage-1 `+=` makes gradients *accumulate*; without clearing,
    this step's gradients would pile on top of last step's and corrupt the update.

**Why an activation at all?** Because stacking purely linear layers is pointless: a linear
map of a linear map is still just one linear map (a product of matrices is another matrix),
so a net with no activations could only ever draw a **straight** boundary — useless for the
curved moons. The nonlinear `tanh` *between* the linear layers is what lets the network bend
its boundary. That's its entire job: expressiveness. (The activation is part of the network;
it is **not** the loss — that's a separate thing you'll meet in Stage 5.)

**Why is the last layer different?** `Layer` takes `**kwargs` purely to forward a per-neuron
option — `nonlin=True/False` — down to each `Neuron`. The `MLP` builds every layer with
`nonlin=True` (so hidden layers apply `tanh`) **except the last**, which it builds with
`nonlin=False` — **no activation, a plain linear output.** That's deliberate: the final
number is a **score** that must be free to be any real value (Stage 5 reads its sign and its
size). If you squashed it with `tanh`, the output would saturate near ±1 and its gradient
would die — bad for a score. So: `tanh` on the hidden layers, linear on the output.

> Note on `relu`: you implemented `relu` in Stage 3, but **this network doesn't use it as a
> layer activation** — the hidden layers use `tanh` and the output is linear. The `relu`
> you'll see in Stage 5 is part of the *loss*, not a layer. (Using `relu` for the hidden
> layers would also work; `tanh` is just the classic choice for micrograd.)

**Your task.** Implement the four classes in `micrograd/nn.py`.

<details>
<summary>Reference implementation — try the task first, then expand</summary>

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
        return act.tanh() if self.nonlin else act     # hidden: tanh; output: linear
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
                       for i in range(len(nouts))]   # last layer -> nonlin=False (linear)
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

</details>

**Checkpoint.** (`MLP(2,[16,16,1])` has `(2·16+16)+(16·16+16)+(16·1+1) = 337` parameters.)

```bash
pytest -q tests/test_nn.py::test_mlp_shapes_and_params
```

---

## Stage 5 — Train it

This is the stage with the real machine-learning ideas. None of them need math beyond your
course; they're spelled out below, in order, with numbers.

### 5.0 — The big picture: what are we even doing?

You have 100 points in the plane, each tagged with one of two classes (which moon it's on).
You want a function that takes a new point `(x1, x2)` and says which class. **The neural net
is that function.** "Learning" means choosing the net's weights so it classifies the points
correctly.

Here's the whole bridge from your calculus course to this:

1. **The net is a differentiable function.** `MLP` is one big expression `f(point; weights)`
   built from your `+`, `*`, `tanh` ops — so you can take its gradient with respect to every
   weight. *That* is why the calculus course was the prerequisite.
2. **"Classify correctly" has no gradient, so we invent a loss.** Whether a point is right is
   a yes/no — flat, with no slope to follow (more on "flat" in 5.2). So we define a smooth
   number, the **loss**, that measures *how wrong* the net is. Lowering the loss = getting
   less wrong. This is the move that turns classification into a calculus problem.
3. **Training = gradient descent on the loss.** The loss is a function of the weights; its
   gradient points toward *more* loss, so we step the weights the opposite way — exactly the
   gradient descent from the course. `backward()` (what you built) computes that gradient
   cheaply. Repeat forward → loss → backward → step, and the net learns.

Everything below is just filling in the specific choices for step 2.

### 5.1 — From the net's output to a class

The last layer is linear, so each point produces one real number `s` — the **score**. Turn
it into a prediction by its **sign**: `s > 0` → class `+1`, `s < 0` → class `−1`. The
decision boundary is where `s = 0`. Example: scores `s = +2.0, −0.4, +0.1` predict classes
`+1, −1, +1`.

### 5.2 — Why a smooth loss, not "count the mistakes" (what *flat* and *smooth* mean)

Imagine turning one weight like a knob and watching the loss.

- If the loss were "number of misclassified points," nudging a weight usually flips nothing,
  so the count sits at `7, 7, 7, …` then suddenly drops to `6` when one point crosses the
  boundary. That's a **staircase**: flat treads where the slope (derivative) is `0`, with
  vertical jumps. A zero slope gives gradient descent **no direction** ("which way do I turn
  the knob?" → no answer), and the jump isn't a slope either. **Flat = derivative 0 = no
  information.**
- A **smooth** loss changes *gradually* as you turn the knob — e.g. `2.5 → 2.3 → 2.1 → …` —
  and has a definite slope at each point telling you which way lowers it. **Smooth = has a
  usable slope you can follow downhill.** That slope is exactly the gradient descent uses.

So we never optimize the error count directly; we optimize a smooth stand-in for it.

### 5.3 — The loss we use: hinge, which is just your ReLU

Label the two classes as `y = +1` and `y = −1` (why ±1: next section). The loss for one
point is:

```
loss = relu(1 − y·s)
```

That's literally the `relu` you built in Stage 3, applied to one simple expression. Plug in
three points (take `y = +1`):

| score `s` | `y·s` | `1 − y·s` | `relu(…)` = loss | meaning |
|---|---|---|---|---|
| `+2.0` | `+2.0` | `−1.0` | **0** | correct and a full unit clear of the boundary → no penalty |
| `+0.1` | `+0.1` | `+0.9` | **0.9** | *technically* correct but too close → still penalized |
| `−1.5` | `−1.5` | `+2.5` | **2.5** | wrong → big penalty |

So the loss is `0` once a point is correct *by a margin of at least 1*, and grows linearly
otherwise. Its gradient is your ReLU's 0/1 gate: `0` when `y·s > 1` (flat side), and `−y`
otherwise. **Why this shape?** Because correct-but-unsure points (`0 < y·s < 1`) still pay,
training keeps pushing the boundary until points sit a full unit clear on their side — it
carves the widest safe gap between the classes. "Max-margin / SVM" is just a name for that;
there's no extra machinery.

> Activation vs. loss — keep them separate. The `tanh` in Stage 4 lives *inside* the net
> (it makes the net nonlinear). This `relu` lives *outside* the net, in the loss (it measures
> wrongness). Both are differentiable, but they do different jobs.

### 5.4 — Why label the classes ±1 (instead of 1/0)

So that the single product `m = y·s` (the **margin**) carries all the information:

- correct & confident: `y=+1, s=+2` → `y·s = +2`
- correct & unsure: `y=+1, s=+0.1` → `y·s = +0.1`
- wrong: `y=+1, s=−1.5` → `y·s = −1.5`

`y·s > 0` exactly when the prediction is on the correct side, and its size is the
confidence. With `0/1` labels you couldn't form this one clean signal (a `0` label wipes the
product out). It's a bookkeeping choice that makes the hinge formula work — not a deep idea.

### 5.5 — A little L2 regularization (optional safety rail)

Add a penalty on weight size to the loss: `total = data_loss + α·Σ(w²)`, with a small `α`
(e.g. `1e-4`). The `Σw²` is the same sum-of-squares (`wᵀw`) from the quadratic-form identity;
its gradient is `2α·w` per weight — a gentle pull toward zero each step ("weight decay").

**Why add it?** With enough freedom the net can *memorize* the 100 points, including their
noise, drawing a wiggly boundary that's perfect here but bad on new points (**overfitting**).
Preferring small weights keeps the function smoother, so it generalizes better. `α` trades
the two goals off: too large under-fits, too small over-fits. On this clean toy you can even
start with `α = 0` — the net still learns; L2 is a safety rail, not load-bearing.

### 5.6 — The data: what `make_moons` returns

You do **not** build the dataset — `data.py` already implements `make_moons(n, noise, seed)`.
You just call it. It returns a tuple `(X, y)` of two NumPy arrays:

- `X` — shape `(n, 2)`. One **row per point**, each row the point's coordinates `[x1, x2]`.
- `y` — shape `(n,)`. One integer **label per point**, in `{0, 1}` (which moon).

A real draw of 8 points:

```
point 0: (+0.581, -0.423)   y=0
point 1: (+0.033, +1.028)   y=0
 ...
point 7: (+1.582, +0.638)   y=1

y (0/1): [0, 0, 0, 0, 1, 1, 1, 1]
```

So each training example is a pair: a 2-D point `X[i]` and its class `y[i]`. You feed the
point into the net as `[Value(x1), Value(x2)]`, and use the label in the loss — after
remapping `{0,1} → {−1,+1}` with `y = y01 * 2 - 1`, because the hinge loss expects ±1. Call
`make_moons` **once, before the loop**; the loop reuses that fixed `X, y` every step
(regenerating it each iteration would be wrong).

### 5.7 — Your task: the training loop

Write `train.py`'s `main()`: generate the data once, build the model, then loop —
forward-pass every point, build the scalar loss (hinge mean + L2), `model.zero_grad()`,
`loss.backward()`, and step each parameter `p.data -= lr * p.grad`. Reset the gradients
every step (Stage 1: they accumulate). A slowly shrinking learning rate helps it settle.

<details>
<summary>Reference implementation — try the task first, then expand</summary>

```python
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from data import make_moons

def forward_loss(model, X, y, alpha=1e-4):
    preds = [model([Value(xi[0]), Value(xi[1])]) for xi in X]
    losses = [(1 + -yi * pi).relu() for yi, pi in zip(y, preds)]   # hinge = relu(1 - y*s)
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    acc = sum((yi > 0) == (pi.data > 0) for yi, pi in zip(y, preds)) / len(y)
    return data_loss + reg, acc

def main():
    X, y01 = make_moons(n=100, noise=0.10, seed=1)   # called ONCE, before the loop
    y = y01 * 2 - 1                                   # {0,1} -> {-1,+1}
    model = MLP(2, [16, 16, 1])
    for k in range(100):
        total, acc = forward_loss(model, X, y)
        model.zero_grad()
        total.backward()
        lr = 1.0 - 0.9 * k / 100                      # simple learning-rate decay
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0:
            print(f"step {k:3d}   loss {total.data:.4f}   acc {acc*100:.1f}%")
```

</details>

### 5.8 — How you'll know it works

Three independent checks, weakest to strongest:
1. **Gradient check (Stage 3)** already proved the engine's math.
2. **`pytest -q` green** — in particular `tests/test_nn.py::test_mlp_learns_moons`, which
   trains your `MLP` (with this exact loss, inline — it does *not* call your `train.py`) and
   asserts accuracy ≥ 90%.
3. **`python train.py`** — watch the printed accuracy climb toward ~100%. (The decision-
   boundary picture comes in Stage 6.)

**Checkpoint.**

```bash
pytest -q tests/test_nn.py::test_mlp_learns_moons
python train.py
```

> Pure-Python scalar autograd is **slow** — one step builds tens of thousands of `Value`
> objects. That slowness is the whole motivation for the *next* capstone: a vectorized engine
> on numpy arrays, where a whole layer is one matrix multiply.

---

## Stage 6 — See the decision boundary

Now *add* a plotting block to your `main()` (it is not pre-written — `train.py` is yours).
Sweep a grid of points over the plane, classify each with the trained model, and shade the
two regions; `python train.py` then writes `boundary.png`. Compare it to
`docs/target_boundary.png` — two cleanly separated moons with a curved boundary between them
is your end-to-end "it learned" confirmation.

<details>
<summary>Reference implementation — full version also in <code>SOLUTION/train.py</code></summary>

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

</details>

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
2. **Swap the hidden activation.** Rebuild the MLP with `relu` hidden layers instead of
   `tanh`; confirm it still separates the moons. (Shows the activation choice is yours.)
3. **Hold out a test set.** Generate two separate `make_moons` draws (different seeds), train
   on one and report accuracy on the other — the honest generalization check. Watch how a
   too-large net or too-small `α` makes train and test accuracy diverge.
4. **Visualize the graph.** Use `graphviz` to draw a small expression's DAG with each node's
   data and grad — you'll *see* the reverse sweep.
5. **Sanity-check against PyTorch.** Rebuild one expression with `torch` tensors
   (`requires_grad=True`), call `.backward()`, and confirm the gradients match yours.

## How the code maps to the math

| operation (code) | local derivative | from the module |
|---|---|---|
| `a + b` | `1`, `1` | sum node passes gradient through |
| `a * b` | `b`, `a` | product rule entries |
| `a ** k` | `k·a**(k-1)` | power rule |
| `relu(x)` | `1` if `x>0` else `0` | diagonal activation Jacobian / 0-1 gate |
| `tanh(x)` | `1 − tanh²x` | elementwise map → diagonal |
| `hinge = relu(1 − y·s)` | `0` if `y·s>1` else `−y` | a ReLU applied to an affine score |
| `α·Σ w²` (L2) | `2α·w` per weight | quadratic-form gradient `∇(wᵀw)=2w` |
| `backward()` | product of locals, reversed | chain rule, swept right-to-left from the scalar loss |
