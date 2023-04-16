import sys
import test
import numpy as np
positions = np.ones((3,5))
position1 = np.zeros((3,2))
velocities = np.random.rand(3,5)
print(positions.shape,position1.shape,velocities.shape)
print(test.push.__doc__)
test.push(positions,velocities,position1,0.1)
print(positions)
