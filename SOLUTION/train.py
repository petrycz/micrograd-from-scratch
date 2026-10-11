"""Train the from-scratch MLP on two-moons, print loss/accuracy, save a decision boundary."""
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP
from data import make_moons

def forward_loss(model, X, y, alpha=1e-4):
    inputs = [[Value(xi[0]), Value(xi[1])] for xi in X]
    preds = [model(xi) for xi in inputs]
    # SVM max-margin loss: hinge( 1 - y*pred ), y in {-1,+1}
    losses = [(1 + -yi * pi).relu() for yi, pi in zip(y, preds)]
    data_loss = sum(losses) * (1.0 / len(losses))
    reg = alpha * sum((p * p for p in model.parameters()), Value(0.0))
    total = data_loss + reg
    acc = sum((yi > 0) == (pi.data > 0) for yi, pi in zip(y, preds)) / len(y)
    return total, acc

def main():
    X, y01 = make_moons(n=100, noise=0.10, seed=1)
    y = y01 * 2 - 1                      # {0,1} -> {-1,+1}
    model = MLP(2, [16, 16, 1])
    for k in range(100):
        total, acc = forward_loss(model, X, y)
        model.zero_grad()
        total.backward()
        lr = 1.0 - 0.9 * k / 100
        for p in model.parameters():
            p.data -= lr * p.grad
        if k % 10 == 0 or k == 99:
            print(f"step {k:3d}   loss {total.data:.4f}   acc {acc*100:.1f}%")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h = 0.25
        xx, yy = np.meshgrid(np.arange(X[:,0].min()-1, X[:,0].max()+1, h),
                             np.arange(X[:,1].min()-1, X[:,1].max()+1, h))
        pts = np.c_[xx.ravel(), yy.ravel()]
        Z = np.array([model([Value(p[0]), Value(p[1])]).data > 0 for p in pts]).reshape(xx.shape)
        plt.figure(figsize=(5,5))
        plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3)
        plt.scatter(X[:,0], X[:,1], c=y01, cmap="RdBu", edgecolors="k", s=28)
        plt.title("micrograd MLP — learned decision boundary")
        plt.tight_layout(); plt.savefig("boundary.png", dpi=110)
        print("saved boundary.png")
    except Exception as e:
        print("plot skipped:", e)

if __name__ == "__main__":
    main()
