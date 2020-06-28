import numpy as np
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D

nx = 41
ny = 41
a = []
b = []
c = []
with open('~/hpc_lecture/13_pde/uvp.txt', 'r') as f:
    s = f.read()
    d = s.split()
    for i in range(ny):
        a.append([float(x) for x in d[nx*i:nx*i+nx]])
        b.append([float(x) for x in d[nx*i+ny*nx:nx*i+nx+ny*nx]])
        c.append([float(x) for x in d[nx*i+2*ny*nx:nx*i+nx+2*ny*nx]])
u = np.array(a)
v = np.array(b)
p = np.array(c)
#print(p[0, :])

x = np.linspace(0,2,nx)
y = np.linspace(0,2,ny)
X, Y = np.meshgrid(x,y)

fig = pyplot.figure(figsize=(11, 7), dpi=100)
pyplot.contourf(X, Y, p, alpha=0.5, cmap=cm.viridis)
pyplot.colorbar()
pyplot.contour(X, Y, p, cmap=cm.viridis)
pyplot.streamplot(X, Y, u, v)
pyplot.xlabel('X')
pyplot.ylabel('Y');
pyplot.show()