import numpy as np
import matplotlib.pyplot as plt

# Fluid Parameters
mu = 0.000102
rho = 1

# Flow Initial and Boundary Conditions (u = u(x,y,t))
R = 0.05 # Radius of pipe (m)
dp_dx = -100 # Inlet pressure gradient in x-direction (Pa/m)
dp_dy = 0 # Inlet pressure gradient in y-direction (Pa/m)
u_max = -dp_dx * R**2 / (4 * mu * rho) # Maximum velocity in the pipe (m/s)

# Boundary Conditions
# No-slip condition at the walls: u(-R,-R,t) = u(R,R,t) = 0

# Initial condition
# Inlet condition: u(x,y,0) = U0 for -R <= y <= R

# Numerical Parameters
L = 3# Length of pipe (m)
N = 200 # Number of spatial points
dx = (2*R)/(N-1) # Spatial step size in x and y directions
dy = dx
dt = 0.001# Time step size
Re = (rho * u_max * 2*R) / mu # Reynolds number

# Governing Equation: Navier-Stokes equation for incompressible flow
# rho*du/dt + u*du/dx  + v*du/dy = - dp/dx + mu*(d^2u/dx^2 + d^2u/dy^2)
# rho*dv/dt + u*dv/dx  + v*dv/dy = - dp/dy + mu*(d^2v/dx^2 + d^2v/dy^2)


Lx = 2.0   # length
Ly = 0.2   # height
Nx = 200
Ny = 50
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
X, Y = np.meshgrid(x, y, indexing='ij')
mask = np.zeros((Nx, Ny), dtype=bool)
# block obstacle
mask[80:85, 10:40] = True

# Discretization using finite difference method
u = np.zeros((Nx, Ny)) # Velocity field in x-direction
v = np.zeros((Nx, Ny)) # Velocity field in y-direction
u[:, :] = 0 # Initial velocity field
v[:, :] = 0 # Initial velocity field
nu = mu / rho
# allocate once (outside loop ideally)
p = np.zeros((Nx, Ny))

for n in range(100):  # increase for better results

    u_old = u.copy()
    v_old = v.copy()

    u_star = np.zeros_like(u)
    v_star = np.zeros_like(v)

    # Tentative velocity

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):

            if not mask[i, j]:  # fluid only

                du_dx = (u_old[i+1,j] - u_old[i-1,j])/(2*dx)
                du_dy = (u_old[i,j+1] - u_old[i,j-1])/(2*dy)

                dv_dx = (v_old[i+1,j] - v_old[i-1,j])/(2*dx)
                dv_dy = (v_old[i,j+1] - v_old[i,j-1])/(2*dy)

                lap_u = (
                    (u_old[i+1,j] - 2*u_old[i,j] + u_old[i-1,j]) / dx**2 +
                    (u_old[i,j+1] - 2*u_old[i,j] + u_old[i,j-1]) / dy**2
                )

                lap_v = (
                    (v_old[i+1,j] - 2*v_old[i,j] + v_old[i-1,j]) / dx**2 +
                    (v_old[i,j+1] - 2*v_old[i,j] + v_old[i,j-1]) / dy**2
                )

                u_star[i,j] = u_old[i,j] + dt * (
                    - u_old[i,j]*du_dx
                    - v_old[i,j]*du_dy
                    + nu * lap_u
                    - dp_dx / rho
                )

                v_star[i,j] = v_old[i,j] + dt * (
                    - u_old[i,j]*dv_dx
                    - v_old[i,j]*dv_dy
                    + nu * lap_v
                )

    # build RHS for pressure Poisson equation
    rhs = np.zeros_like(p)

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if not mask[i,j]:
                rhs[i,j] = (rho/dt) * (
                    (u_star[i+1,j] - u_star[i-1,j])/(2*dx) +
                    (v_star[i,j+1] - v_star[i,j-1])/(2*dy)
                )

    # solve pressure Poisson
    for k in range(230):  # iterations

        p_old = p.copy()

        for i in range(1, Nx-1):
            for j in range(1, Ny-1):
                if not mask[i,j]:
                    p[i,j] = (
                        (p_old[i+1,j] + p_old[i-1,j]) * dy**2 +
                        (p_old[i,j+1] + p_old[i,j-1]) * dx**2
                        - rhs[i,j]*dx**2*dy**2
                    ) / (2*(dx**2 + dy**2))

        # pressure boundary conditions
        p[:,0]  = p[:,1]
        p[:,-1] = p[:,-2]
        p[0,:]  = p[1,:]
        p[-1,:] = p[-2,:]
        p[mask] = 0

    # velocity correction

    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            if not mask[i,j]:

                u[i,j] = u_star[i,j] - dt/rho * (p[i+1,j] - p[i-1,j])/(2*dx)
                v[i,j] = v_star[i,j] - dt/rho * (p[i,j+1] - p[i,j-1])/(2*dy)

    # boundary conditions

    # obstacle (no-slip)
    u[mask] = 0
    v[mask] = 0

    # inlet
    u[0,:] = 4*(y*(Ly - y)) / Ly**2
    v[0,:] = 0.0

    # outlet (zero gradient)
    u[-1,:] = u[-2,:]
    v[-1,:] = v[-2,:]

    # walls
    u[:,0] = 0
    u[:,-1] = 0
    v[:,0] = 0
    v[:,-1] = 0

    # small perturbation for vortix generation
    if n == 0:
        u += 0.01*np.random.randn(Nx, Ny)
    else:
        u += 0.001*np.random.randn(Nx, Ny)

    # Active time step adjustment for stability (CFL condition)
    dt = 0.2 * min(dx**2/nu, dx/(np.max(np.sqrt(u**2+v**2))+1e-8))
    
# Plotting the velocity field
V = np.sqrt(u**2 + v**2)


plt.figure(figsize=(10,3))
plt.contourf(X, Y, V, levels=50, cmap='jet')
plt.colorbar()
plt.streamplot(x, y, u.T, v.T, color='k', density=1.5)
plt.title("Velocity field with streamlines")
plt.show()
