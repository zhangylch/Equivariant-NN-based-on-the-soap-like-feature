import sys
import test
import numpy as np
positions = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
position1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
velocities = np.array([[0, 1, 2], [0, 3, 2], [0, 1, 3]])
test.push(positions,velocities,position1,0.1)
print(positions)
