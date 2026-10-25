"""
train.py — train your MLP on two-moons and draw the decision boundary (GUIDE Stages 5-6).

Build this once your engine and nn pass their tests. The reference version lives
in SOLUTION/train.py if you get stuck.
"""
from data import make_moons
from micrograd.nn import MLP
from micrograd.engine import Value
def get_loss(score, y):
    inp = Value(1)- score*y
    outp = inp.relu()
    return outp
    

def main():
    step_size = 0.50
    reps = 100
    # generate the data
    X, y = make_moons(100)
    #remap
    y = [-1 if ys == 0 else 1 for ys in y]
    # build graph (mlp)
    mlp = MLP(2, [13,13,1])
    for rep in range(reps):
        step_size = max(0.1, step_size-0.01) if rep%10==0 else step_size
        loss = 0
    # for each point x,y
        for i, j in zip(X, y):
            #print("i,j :", i ,j )
            res = mlp(i)
        
            loss += get_loss(res, j)
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data -= p.grad*step_size*1/len(X)

        print("loss:", loss.data)
    
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")                 # writes a file; drop this line for a popup window
    import matplotlib.pyplot as plt

    # a fresh batch, different seed => points the model never trained on
    Xt, yt01 = make_moons(n=100, noise=0.10, seed=1)
    yt = [-1 if v == 0 else 1 for v in yt01]

    # accuracy on the new batch
    preds = [mlp([Value(p[0]), Value(p[1])]).data for p in Xt]
    acc = sum((pr > 0) == (t > 0) for pr, t in zip(preds, yt)) / len(yt)
    print(f"accuracy on new batch: {acc*100:.1f}%")

    # decision boundary: classify a grid of points and shade the two regions
    pad, h = 1.0, 0.25
    x_min, x_max = Xt[:,0].min()-pad, Xt[:,0].max()+pad
    y_min, y_max = Xt[:,1].min()-pad, Xt[:,1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([mlp([Value(px), Value(py)]).data > 0 for px, py in grid]).reshape(xx.shape)

    plt.figure(figsize=(5, 5))
    plt.contourf(xx, yy, Z, cmap="RdBu", alpha=0.3)                 # the learned regions
    plt.scatter(Xt[:, 0], Xt[:, 1], c=yt01, cmap="RdBu", edgecolors="k", s=28)  # the new points
    plt.title(f"new batch — {acc*100:.0f}% correct")
    plt.tight_layout()
    plt.savefig("new_batch.png", dpi=110)
    print("saved new_batch.png")
    # forward pass - res = __call__(x) 
    # calculate the loss (res, y)
    # mlp.zero_grad()
    # loss.backward()
    # for p in mlp.parameters() p.data -= p.grad*step_size


if __name__ == "__main__":
    main()
