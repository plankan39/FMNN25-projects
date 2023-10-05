import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from pprint import pprint


################ Describing the problem and setting parameters ################

# Only rectangular rooms are allowed with constant boundary along each side

# Describe the length of the sides of the rectangle
x_len = 1.0
y_len = 1.0

# Number of discrete points along the x-axis
x_N_tot = 30

# The stepsize for the discrete points
h = x_len / x_N_tot
print(f"h={h}")

# Number of discrete points along y-axis
y_N_tot = int(y_len / h)

# As we know the boundary points the number of unknown points along x and y are
x_N = x_N_tot - 2
y_N = y_N_tot - 2

# Total number of unknown points
N = x_N * y_N

print(f"x_N = {x_N}, y_N = {y_N}, N = {N}")

# Boundaries
top_b = [40] * x_N_tot
bot_b = [5] * x_N_tot
left_b = [15] * y_N_tot
right_b = [15] * y_N_tot

print(f"left: {left_b}")

######################### Constructing Laplace matrix #########################

# Main diagonal
d_0 = -4 * np.ones(N)

# The agacent diagonals under and over d_0
d_1 = np.ones(N - 1)
# Change the points where Boundary conditions are imposed
d_1[[k * y_N - 1 for k in range(1, x_N)]] = 0

# The two last diagonals that are y_N
d_y_N = np.ones(N - y_N)

# The Laplace matrix
A = sp.sparse.diags([d_0, d_1, d_1, d_y_N, d_y_N], [0, 1, -1, y_N, -y_N], format="csc")
print(f"A:\n{A.toarray()}")
# The known boundary conditions
b = np.zeros(N)

b[-y_N:] -= right_b[1:-1]
b[: y_N] -= left_b[1:-1]
b[::y_N] -= bot_b[1:-1]
b[y_N - 1 :: y_N] -= top_b[1:-1]
print(f"b:\n{b}")
theta = sp.sparse.linalg.spsolve(A, b)

# print(theta)

X, Y = np.meshgrid(np.linspace(0, x_len, x_N_tot), np.linspace(0, y_len, y_N_tot))

T = np.zeros_like(X)


T[-1, :] = top_b
T[0, :] = bot_b
T[:, 0] = left_b
T[:, -1] = right_b

for j in range(1, y_N_tot - 1):
    for i in range(1, x_N_tot - 1):
        T[j, i] = theta[j + (i - 1) * y_N - 1]

print(f"T:\n{T}")

# print(T)

plt.title("Temperature in the room")
plt.contourf(X, Y, T, levels=1000, cmap=plt.cm.coolwarm)

plt.colorbar()

plt.show()
