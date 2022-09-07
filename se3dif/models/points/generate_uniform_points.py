import numpy as np
import matplotlib.pyplot as plt



pts = (np.random.rand(1000,3) - .5)*2.
print(pts.shape)

file = 'UniformPts.npy'
np.save(file, pts)


## Figure ##
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(pts[:,0], pts[:,1], pts[:,2])
plt.show()
