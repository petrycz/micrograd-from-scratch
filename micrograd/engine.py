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
        # Stage 1 — store: data (float), grad (0.0), _backward (a no-op closure),
        # _prev (set of child Values), _op (a label string, for debugging).
        raise NotImplementedError("Stage 1: see GUIDE.md")

    def __add__(self, other):
        raise NotImplementedError("Stage 1")

    def __mul__(self, other):
        raise NotImplementedError("Stage 1")

    def backward(self):
        raise NotImplementedError("Stage 2")

    # ---- Stage 3: add __pow__, relu, tanh, exp, and the conveniences below ----
    # __neg__, __radd__, __sub__, __rsub__, __rmul__, __truediv__, __rtruediv__, __repr__
