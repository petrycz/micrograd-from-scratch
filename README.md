# micrograd-from-scratch

A tiny **reverse-mode automatic differentiation** engine and neural-network library,
built from nothing but Python scalars — then used to train a multilayer perceptron
that learns a non-linear decision boundary.

This is the engine underneath PyTorch/TensorFlow, stripped to its essence: a scalar
`Value` that records how it was computed, and a `.backward()` that walks the graph in
reverse applying the chain rule. ~150 lines, no dependencies for the core.

![decision boundary](docs/target_boundary.png)

*The MLP above reaches 100% accuracy on two interleaving moons — trained entirely by
gradients produced by this engine.*

## Quickstart

```bash
git clone <your-repo-url> && cd micrograd-from-scratch
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q          # run the test suite
python train.py    # train the MLP and save boundary.png
```

## What's inside

| path | what it is |
|------|------------|
| `micrograd/engine.py` | the `Value` autograd core — forward ops + `backward()` |
| `micrograd/nn.py`     | `Neuron`, `Layer`, `MLP` built on `Value` |
| `train.py`            | trains on two-moons, draws the decision boundary |
| `data.py`             | toy two-moons dataset (numpy) |
| `tests/`              | pytest suite: op correctness, finite-difference gradient checks, a learns-the-data test |
| `GUIDE.md`            | **the step-by-step build guide** — start here |
| `SOLUTION/`           | reference implementation (peek only if stuck) |

## How to build it

This repo is a guided project. Open **[GUIDE.md](GUIDE.md)** and work through the
stages; the tests turn green one group at a time as you implement each piece.

## Credits

The scalar-autograd design follows Andrej Karpathy's
[micrograd](https://github.com/karpathy/micrograd). This version is rebuilt from
scratch as a capstone for the matrix-calculus module of a self-study calculus course,
with a test suite and a guided build.

## License

MIT — see [LICENSE](LICENSE).
