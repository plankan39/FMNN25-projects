
    def calculate_hessian(self, H, x_next, x, g, g_next):
        s = x_next - x
        y = g_next - g

        ys_t = np.outer(y, s)
        denominator = np.dot(s, y)

        if denominator == 0:
            raise Exception("Denominator is zero in Bad Broyden update.")

        H += ys_t / denominator

        return H