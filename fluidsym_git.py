import numpy as numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot, cm
from matplotlib.colors import Normalize

nx = 41
ny = 41

dx = 2 / float(nx-1)
dy = 2 / float(ny-1)

u = numpy.ones((ny,nx))
v = numpy.ones((ny,nx))

sigma = 0.001
nu = 0.01
dt = sigma * dx * dy / nu

def equation_of_motion(u, v, dt, dx, dy, nu):
    un = u.copy()
    vn = v.copy()
    u[1:-1, 1:-1] = (un[1:-1, 1:-1] -
                        dt/dx * un[1:-1, 1:-1] *
                        (un[1:-1, 1:-1] - un[1:-1,0:-2]) -
                        dt/dy * vn[1:-1, 1:-1] *
                        (un[1:-1, 1:-1] - un[0:-2, 1:-1]) +
                        nu * dt / dx**2 *
                        (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
                        nu * dt / dy **2 *
                        (un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2,1:-1]))

    v[1:-1, 1:-1] = (vn[1:-1, 1:-1] -
                    dt/dx * un[1:-1, 1:-1] *
                    (vn[1:-1, 1:-1] - vn[1:-1,0:-2]) -
                    dt / dy * vn[1:-1, 1:-1] *
                    (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) +
                    nu * dt / dx**2 *
                    (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
                    nu * dt / dy **2 *
                    (vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2,1:-1]))

    return (u,v)

def boundary(u, v, nozzle_u, nozzle_v, nx, ny, t_step):
   u[0,:] = 0
   u[-1,:] = 0
   u[:,0] = 0
   u[:,-1] = 0

   v[0,:] = 0
   v[-1,:] = 0
   v[:,0] = 0
   v[:,-1] = 0

   #special nozzle BC
   u[ny//2-2:ny//2+2, 0] = nozzle_u[t_step]
   v[ny//2-2:ny//2+2, 0] = nozzle_v[t_step]

   return (u,v)

nt = 2510
f = None

initial_u = numpy.zeros((nx,ny))
initial_v = numpy.zeros((ny,ny))
nozzle_u = numpy.append(10*numpy.ones(1000), numpy.zeros(nt))
nozzle_v = numpy.append(10*numpy.ones(1000), numpy.zeros(nt))

def simulate(f,u,v,dt,dx,dy,nu,steps, nozzle_u, nozzle_v, nx, ny):
    for i in range(steps):
        (u,v) = equation_of_motion(u, v, dt, dx, dy, nu)
        (u,v) = boundary(u, v, nozzle_u, nozzle_v, nx, ny, i)
        ax = pyplot.figure()
        norm = Normalize()
        magnitude = numpy.sqrt(u[::2]**2 + v[::2]**2)
        pyplot.quiver(u,v, norm(magnitude), scale = 60, cmap = pyplot.cm.jet)
        ax.savefig('frame'.zfill(5)+str(i)+'.png', dpi=300)
        ax.clear()



simulate(f,initial_u,initial_v,dt,dx,dy,nu,nt, nozzle_u, nozzle_v, nx, ny)
pyplot.close()
