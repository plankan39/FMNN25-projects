import numpy as np

from solve_single_room_2 import Boundary

TEST = 2

# Tunable parameters
WALL = 15
HEATER = 40
WINDOW = 5
W = 0.8
ITERATIONS = 10

# Describing room 1
X_N_1 = 4
Y_N_1 = X_N_1
# Describing room 2
X_N_2 = X_N_1
Y_N_2 = 2 * X_N_2
# Describing room 3
X_N_3 = X_N_1
Y_N_3 = Y_N_1

# The
H = 1 / (X_N_1 - 1)
