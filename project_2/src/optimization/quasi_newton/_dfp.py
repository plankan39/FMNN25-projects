import numpy as np


def update_inv_hessian(
    x_next: np.ndarray,
    x: np.ndarray,
    g_next: np.ndarray,
    g: np.ndarray,
    h_inv: np.ndarray,
):
    dx = x_next - x
    dg = g_next - g

    term1 = np.outer(dx, dx) / dx.T.dot(dg)
    
    w2 = h_inv.dot(dg) # See line 19 of the algorithm
    term2 = np.outer(w2, w2) / w2.T.dot(dg)
    
    dh = term1 - term2 # See line 22 of the algorithm
    
    h_inv += dh


h = np.eye(2)

x_next = np.array([1,1])
x = np.array([0.5,0.5])


g_next = np.array([0.2,0.2])
g = np.array([0.5,0.5])
print(h)
update_inv_hessian(x_next, x, g_next, g, h)
print(h)

