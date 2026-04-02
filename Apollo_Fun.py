import numpy as np 
import matplotlib.pyplot as plt
mu = 0.012277471
mup = 1 - mu
x0 = 0.994
xp0 = 0
y0 = 0
yp0 = -2.00158510637908252240537862224

def f(z,w,x,y,t):
    return x+2*z -mup*(x+mu)/((x+mu)**2+y**2)**(3/2) - mu*(x-mup)/((x-mup)**2+y**2)**(3/2)
def g(z,w,x,y,t):
    return y-2*w -mup*y/((x+mu)**2+y**2)**(3/2) - mu*y/((x-mup)**2+y**2)**(3/2)
dh = 0.001
t = np.arange(2000,2020,dh)
z = np.zeros(len(t))
w = np.zeros(len(t))
x = np.zeros(len(t))
y = np.zeros(len(t))
zE = np.zeros(len(t))
wE = np.zeros(len(t))
xE = np.zeros(len(t))
yE = np.zeros(len(t))
zH = np.zeros(len(t))
wH = np.zeros(len(t))
xH = np.zeros(len(t))
yH = np.zeros(len(t))
z[0] = yp0
w[0] = xp0
x[0] = x0
y[0] = y0
zH[0] = yp0
wH[0] = xp0
xH[0] = x0
yH[0] = y0
zE[0] = yp0
wE[0] = xp0
xE[0] = x0
yE[0] = y0
for i in range(0,len(t)-1):
    zE[i+1] = zE[i] + g(zE[i],wE[i],xE[i],yE[i],t[i])*dh
    wE[i+1] = wE[i] + f(zE[i],wE[i],xE[i],yE[i],t[i])*dh
    xE[i+1] = xE[i] + wE[i]*dh
    yE[i+1] = yE[i] + zE[i]*dh
def rk4(z,w,x,y,t,dh):
    k1 = [g(z,w,x,y,t),f(z,w,x,y,t),w,z]
    k2 = [g(z+0.5*dh*k1[0],w+0.5*dh*k1[1],x+0.5*dh*k1[2],y+0.5*dh*k1[3],t+0.5*dh),f(z+0.5*dh*k1[0],w+0.5*dh*k1[1],x+0.5*dh*k1[2],y+0.5*dh*k1[3],t+0.5*dh),w+0.5*dh*k1[1],z+0.5*dh*k1[0]]
    k3 = [g(z+0.5*dh*k2[0],w+0.5*dh*k2[1],x+0.5*dh*k2[2],y+0.5*dh*k2[3],t+0.5*dh),f(z+0.5*dh*k2[0],w+0.5*dh*k2[1],x+0.5*dh*k2[2],y+0.5*dh*k2[3],t+0.5*dh),w+0.5*dh*k2[1],z+0.5*dh*k2[0]]
    k4 = [g(z+dh*k3[0],w+dh*k3[1],x+dh*k3[2],y+dh*k3[3],t+dh),f(z+dh*k3[0],w+dh*k3[1],x+dh*k3[2],y+dh*k3[3],t+dh),w+dh*k3[1],z+dh*k3[0]]
    z_next = z + (dh/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0])
    w_next = w + (dh/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])
    x_next = x + (dh/6)*(k1[2]+2*k2[2]+2*k3[2]+k4[2])
    y_next = y + (dh/6)*(k1[3]+2*k2[3]+2*k3[3]+k4[3])
    return z_next,w_next,x_next,y_next
for i in range(0,len(t)-1):
    z[i+1],w[i+1],x[i+1],y[i+1] = rk4(z[i],w[i],x[i],y[i],t[i],dh)
def heun(zH, wH,xH,yH):
    k1 = [g(zH,wH,xH,yH,t[i]),f(zH,wH,xH,yH,t[i]),wH,zH]
    k2 = [g(zH+dh*k1[0],wH+dh*k1[1],xH+dh*k1[2],yH+dh*k1[3],t[i]+dh),f(zH+dh*k1[0],wH+dh*k1[1],xH+dh*k1[2],yH+dh*k1[3],t[i]+dh),wH+dh*k1[1],zH+dh*k1[0]]
    z_next = zH + (dh/2)*(k1[0]+k2[0])
    w_next = wH + (dh/2)*(k1[1]+k2[1])
    x_next = xH + (dh/2)*(k1[2]+k2[2])
    y_next = yH + (dh/2)*(k1[3]+k2[3])
    return z_next,w_next,x_next,y_next
for i in range(0,len(t)-1):
    zH[i+1],wH[i+1],xH[i+1],yH[i+1] = heun(zH[i],wH[i],xH[i],yH[i])
plt.plot(x,y, 'blue', label='RK4')
plt.plot(xE,yE, 'red', label='Euler')
plt.plot(xH,yH, 'green', label='Heun')
plt.xlabel('x (AU)')
plt.ylabel('y (AU)')
plt.title('Trajectory of the Apollo spacecraft using RK4, Heun and Euler methods')
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True, borderpad=1)
plt.show()