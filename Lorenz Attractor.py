import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # enables 3D plotting
x0 = 2.3
y0 = 6.7
z0 = 2
Pr = 10
Ra = 28
beta = 8/3
def f(x,y,z):
    return [Pr*(y-x), x*(Ra-z)-y, x*y-beta*z]
dt = 0.01
t = np.arange(0,30,dt)
# Range-Kutta 14th order method
x = np.zeros(len(t))
y = np.zeros(len(t))
z = np.zeros(len(t))
x[0] = x0
y[0] = y0
z[0] = z0
# Euler method
xE = np.zeros(len(t))
yE = np.zeros(len(t))
zE = np.zeros(len(t))
xE[0] = x0
yE[0] = y0
zE[0] = z0
# Heun Method
xH = np.zeros(len(t))
yH = np.zeros(len(t))
zH = np.zeros(len(t))
xH[0] = x0
yH[0] = y0
zH[0] = z0
def rk4(x,y,z,t):
    k1 = f(x,y,z)
    k2 = f(x+0.5*dt*k1[0], y+0.5*dt*k1[1], z+0.5*dt*k1[2])
    k3 = f(x+0.5*dt*k2[0], y+0.5*dt*k2[1], z+0.5*dt*k2[2])
    k4 = f(x+dt*k3[0], y+dt*k3[1], z+dt*k3[2])
    x_next = x + (dt/6)*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
    y_next = y + (dt/6)*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
    z_next = z + (dt/6)*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
    return x_next, y_next, z_next
for i in range(0,len(t)-1):
    x[i+1],y[i+1],z[i+1] = rk4(x[i],y[i],z[i],t[i])
def euler(x,y,z,t):
    k1 = f(x,y,z)
    x_next = x + dt*k1[0]
    y_next = y + dt*k1[1]
    z_next = z + dt*k1[2]
    return x_next, y_next, z_next
for i in range(0,len(t)-1):
    xE[i+1],yE[i+1],zE[i+1] = euler(xE[i],yE[i],zE[i],t[i])
def heun(x,y,z,t):
    k1 = f(x,y,z)
    x_predict = x + dt*k1[0]
    y_predict = y + dt*k1[1]
    z_predict = z + dt*k1[2]
    k2 = f(x_predict, y_predict, z_predict)
    x_next = x + (dt/2)*(k1[0] + k2[0])
    y_next = y + (dt/2)*(k1[1] + k2[1])
    z_next = z + (dt/2)*(k1[2] + k2[2])
    return x_next, y_next, z_next
for i in range(0,len(t)-1):
    xH[i+1],yH[i+1],zH[i+1] = heun(xH[i],yH[i],zH[i],t[i])
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x,y,z, 'blue')
ax.plot(xE,yE,zE, 'red')
ax.plot(xH,yH,zH, 'green')
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_zlabel('Z-axis')
ax.set_title('Lorenz Attractor')
plt.show()