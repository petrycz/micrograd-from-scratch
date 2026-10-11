from micrograd.engine import Value
from micrograd.nn import MLP
from data import make_moons

def test_mlp_shapes_and_params():
    model = MLP(2, [16, 16, 1])
    out = model([Value(0.5), Value(-0.5)])
    assert isinstance(out, Value)
    # (2*16+16) + (16*16+16) + (16*1+1) = 337
    assert len(model.parameters()) == 337

def test_mlp_learns_moons():
    # small + short so the suite stays fast on a laptop; still clearly learns
    X, y01 = make_moons(n=60, noise=0.10, seed=1)
    y = y01 * 2 - 1
    model = MLP(2, [16, 16, 1])
    def loss():
        preds = [model([Value(p[0]), Value(p[1])]) for p in X]
        losses = [(1 + -yi * pi).relu() for yi, pi in zip(y, preds)]
        dl = sum(losses) * (1.0 / len(losses))
        reg = 1e-4 * sum((p * p for p in model.parameters()), Value(0.0))
        acc = sum((yi > 0) == (pi.data > 0) for yi, pi in zip(y, preds)) / len(y)
        return dl + reg, acc
    for k in range(40):
        total, acc = loss()
        model.zero_grad(); total.backward()
        lr = 1.0 - 0.9 * k / 40
        for p in model.parameters():
            p.data -= lr * p.grad
    _, acc = loss()
    assert acc >= 0.90, f"only reached {acc}"
