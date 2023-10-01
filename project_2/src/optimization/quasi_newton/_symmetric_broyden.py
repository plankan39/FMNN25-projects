
def calculate_hessian(self, H, x_next, x, g, g_next):
    s = x_next - x
    y = g_next - g

    ys_t = np.outer(y, s)
    sy_t = np.outer(s, y)
    denominator1 = np.dot(s, y)
    denominator2 = np.dot(y, s - np.dot(H, y))

    if denominator1 == 0 or denominator2 == 0:
        raise Exception("Denominator is zero in Symmetric Broyden update.")

    H += ys_t / denominator1 - np.dot(ys_t, sy_t) / denominator2

    return H