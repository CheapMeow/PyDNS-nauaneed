import numpy as np
from matplotlib import pyplot

import os
print(os.getcwd())

import sys
sys.path.append(os.getcwd())

##variable declarations
nx = 512
ny = 512
lx = 8
ly = 8
dx = lx / (nx - 1)
dy = ly / (ny - 1)
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
xx, yy = np.meshgrid(x, y)
#nt = 220000
nt = 100000
saveth_iter = 500
save_start = 1

##physical variables
Re = 100
rho = 1
R = 0.15
nu = rho * 1 * R * 2 / Re
F = 0
dt = 1.8e-3

# boundary conditions
bc = {'x': 'neumann', 'y': 'free-slip'}

time = 4.5
index = 10

p = np.genfromtxt("./data/p_t=%.4f_%d.csv" % (time, index), delimiter=",")
u = np.genfromtxt("./data/u_t=%.4f_%d.csv" % (time, index), delimiter=",")
v = np.genfromtxt("./data/v_t=%.4f_%d.csv" % (time, index), delimiter=",")

# fig = pyplot.figure(figsize=(11, 7), dpi=100)
# pyplot.quiver(xx[::3, ::3], yy[::3, ::3], u[::3, ::3], v[::3, ::3])

# fig = pyplot.figure(figsize=(11, 7), dpi=100)
# pyplot.quiver(xx, yy, u, v)

import derive

vort = derive.vorticity(u, v, dx, dy, bc)

levels = np.linspace(-4, 4, 100)

fig = pyplot.figure(figsize=(11, 7), dpi=100)
CS = pyplot.contourf(xx[::3, ::3], yy[::3, ::3], vort[::3, ::3], levels)
fig.colorbar(CS)

pyplot.show()
