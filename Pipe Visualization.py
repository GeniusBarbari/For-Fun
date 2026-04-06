import numpy as np
import matplotlib.pyplot as plt

# Load data from the text file
# Coordinates: x (col 0), y (col 1)
# Flow data: u (col 2), v (col 3), p (col 4), mask (col 5)
data = np.loadtxt('pipe_flow_data.txt')

# Grid dimensions provided in the file header
nx, ny = 300, 120

# Reshape the columns into 2D grids (Ny, Nx)
# Transpose or ordering may depend on how the simulation data was written
X = data[:, 0].reshape((nx, ny)).T
Y = data[:, 1].reshape((nx, ny)).T
U = data[:, 2].reshape((nx, ny)).T
V = data[:, 3].reshape((nx, ny)).T
P = data[:, 4].reshape((nx, ny)).T
Mask = data[:, 5].reshape((nx, ny)).T
v = np.sqrt(U**2 + V**2)  # Velocity magnitude for visualization
# Calculate velocity magnitude
velocity_mag = np.sqrt(U**2 + V**2)

# Create the visualization
plt.figure(figsize=(12, 6))

# Plot the horizontal velocity (u) as a filled contour
contour = plt.contourf(X, Y, v, levels=50, cmap='jet')
plt.colorbar(contour, label='Horizontal Velocity (u)')

# Overlay the solid obstacles (mask=1)
# We use a masked array to only show the obstacles
plt.imshow(np.ma.masked_where(Mask == 0, Mask), 
           extent=[X.min(), X.max(), Y.min(), Y.max()], 
           origin='lower', cmap='gray_r', alpha=0.5)

plt.title('Pipe Flow Simulation: Horizontal Velocity Field')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')
plt.axis('equal')
plt.tight_layout()

plt.show()