import math, random
from micrograd.engine import Value

def test_add_mul_forward():
    a, b = Value(2.0), Value(-3.0)
    assert (a + b).data == -1.0
    assert (a * b).data == -6.0

def test_backward_single_neuron():
    # L = (y - t)^2, y = relu(w*x + b)  — the beat's example, numbers checked by hand
    x, w, b, t = Value(1.0), Value(2.0), Value(1.0), 5.0
    z = w * x + b
    y = z.relu()
    L = (y - t) ** 2
    L.backward()
    assert abs(z.data - 3.0) < 1e-9
    assert abs(y.data - 3.0) < 1e-9
    assert abs(L.data - 4.0) < 1e-9          # (3-5)^2
    assert abs(L.grad - 1.0) < 1e-9
    assert abs(y.grad - (2*(3-5))) < 1e-9    # -4
    assert abs(w.grad - (-4 * 1.0)) < 1e-9   # dL/dw = dL/dz * x = -4*1
    assert abs(b.grad - (-4.0)) < 1e-9
    assert abs(x.grad - (-4 * 2.0)) < 1e-9   # dL/dx = dL/dz * w

def _numeric_grad(f, x, eps=1e-6):
    return (f(x + eps) - f(x - eps)) / (2 * eps)

def test_gradient_check_random_expression():
    random.seed(0)
    for _ in range(50):
        av, bv = random.uniform(-2, 2), random.uniform(0.1, 2)
        def expr(a_, b_):
            a, b = Value(a_), Value(b_)
            out = (a * b + a.tanh()) * (b ** 2) + (a + b).relu()
            return a, b, out
        a, b, out = expr(av, bv)
        out.backward()
        ga_num = _numeric_grad(lambda t: expr(t, bv)[2].data, av)
        gb_num = _numeric_grad(lambda t: expr(av, t)[2].data, bv)
        assert abs(a.grad - ga_num) < 1e-4, (a.grad, ga_num)
        assert abs(b.grad - gb_num) < 1e-4, (b.grad, gb_num)

def test_accumulation_reused_node():
    a = Value(3.0)
    b = a + a            # a used twice -> grad should accumulate to 2
    b.backward()
    assert abs(a.grad - 2.0) < 1e-9
