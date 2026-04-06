/*
 * Pipe Flow Simulation in C++
 * Translated from Pipe_Flow.py
 *
 * Simulates 2D incompressible Navier-Stokes flow in a channel with a
 * rectangular obstacle, using the projection (fractional-step) method:
 *   1. Tentative velocity (advection + diffusion + pressure gradient)
 *   2. Pressure Poisson equation (Gauss-Seidel iterations)
 *   3. Velocity correction to enforce divergence-free field
 *
 * Output: velocity_field.ppm  (jet-coloured speed |u| with streamlines)
 *
 * Compile:
 *   g++ -O2 -o pipe_flow pipe_flow.cpp
 * Run:
 *   ./pipe_flow
 */

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>

// Flat 2-D array helpers  (row-major: index = i*Ny + j)

using Array2D = std::vector<double>;
using BoolArr = std::vector<bool>;

static inline double& at(Array2D& a, int i, int j, int Ny)
{ return a[i*Ny + j]; }
static inline double  at(const Array2D& a, int i, int j, int Ny)
{ return a[i*Ny + j]; }

// Write solution data to text file for Python post-processing

static void write_data(const std::string& fname,
                       const Array2D& u, const Array2D& v, const Array2D& p,
                       const std::vector<double>& x, const std::vector<double>& y,
                       const BoolArr& mask,
                       int Nx, int Ny)
{
    std::ofstream f(fname);
    if (!f) { std::cerr << "Cannot open " << fname << "\n"; return; }

    // Header: grid dimensions
    f << "# Pipe flow simulation output\n";
    f << "# Nx=" << Nx << " Ny=" << Ny << "\n";
    f << "# Columns: x  y  u  v  p  mask\n";
    f << "# mask=1 means solid obstacle cell\n";

    f << std::scientific;
    f.precision(8);

    for (int i = 0; i < Nx; ++i)
    for (int j = 0; j < Ny; ++j)
    {
        f << x[i]             << " "
          << y[j]             << " "
          << at(u,i,j,Ny)     << " "
          << at(v,i,j,Ny)     << " "
          << at(p,i,j,Ny)     << " "
          << (mask[i*Ny+j] ? 1 : 0) << "\n";
    }

    std::cout << "Data written to " << fname << "  (" << Nx*Ny << " points)\n";
}


int main()
{
    // Fluid parameters
    const double mu  = 0.000102;
    const double rho = 1.0;
    const double nu  = mu / rho;

    // Geometry parameters
    const double Lx = 2.0;
    const double Ly = 0.2;
    const int    Nx = 300;
    const int    Ny = 120;

    const double dx = Lx*0.001 / (Nx - 1);
    const double dy = Ly*0.001 / (Ny - 1);

    // x, y coordinate arrays
    std::vector<double> x(Nx), y(Ny);
    for (int i = 0; i < Nx; ++i) x[i] = i * dx;
    for (int j = 0; j < Ny; ++j) y[j] = j * dy;

    // Obstacle mask
    BoolArr mask(Nx * Ny, false);
    for (int i = 80; i < 85; ++i)
        for (int j = 10; j < 40; ++j)
            mask[i*Ny + j] = true;

    // Initializing Fields 
    Array2D u(Nx*Ny, 0.0), v(Nx*Ny, 0.0);
    Array2D u_old(Nx*Ny), v_old(Nx*Ny);
    Array2D u_star(Nx*Ny), v_star(Nx*Ny);
    Array2D p(Nx*Ny, 0.0), p_old(Nx*Ny);
    Array2D rhs(Nx*Ny, 0.0);

    // Pressure gradient (driving force) 
    const double dp_dx = -100.0;
    // dp_dy = 0

    // Random-number generator for perturbations 
    std::mt19937 rng(42);
    std::normal_distribution<double> gauss(0.0, 1.0);

    double dt = 0.00001;

    const int N_TIME   = 3000;   // outer time steps
    const int POISSON  = 400;  // pressure iterations per step

    // Time-stepping loop
    for (int n = 0; n < N_TIME; ++n)
    {
        if (n % 10 == 0)
            std::cout << "Time step " << n << "/" << N_TIME
                      << "  dt=" << dt << "\n";

        u_old = u;
        v_old = v;

        std::fill(u_star.begin(), u_star.end(), 0.0);
        std::fill(v_star.begin(), v_star.end(), 0.0);

        // Tentative velocity
        for (int i = 1; i < Nx-1; ++i)
        for (int j = 1; j < Ny-1; ++j)
        {
            if (mask[i*Ny+j]) continue;

            double du_dx = (at(u_old,i+1,j,Ny) - at(u_old,i-1,j,Ny)) / (2*dx);
            double du_dy = (at(u_old,i,j+1,Ny) - at(u_old,i,j-1,Ny)) / (2*dy);
            double dv_dx = (at(v_old,i+1,j,Ny) - at(v_old,i-1,j,Ny)) / (2*dx);
            double dv_dy = (at(v_old,i,j+1,Ny) - at(v_old,i,j-1,Ny)) / (2*dy);

            double lap_u = (at(u_old,i+1,j,Ny) - 2*at(u_old,i,j,Ny) + at(u_old,i-1,j,Ny)) / (dx*dx)
                         + (at(u_old,i,j+1,Ny) - 2*at(u_old,i,j,Ny) + at(u_old,i,j-1,Ny)) / (dy*dy);

            double lap_v = (at(v_old,i+1,j,Ny) - 2*at(v_old,i,j,Ny) + at(v_old,i-1,j,Ny)) / (dx*dx)
                         + (at(v_old,i,j+1,Ny) - 2*at(v_old,i,j,Ny) + at(v_old,i,j-1,Ny)) / (dy*dy);

            at(u_star,i,j,Ny) = at(u_old,i,j,Ny) + dt * (
                - at(u_old,i,j,Ny)*du_dx
                - at(v_old,i,j,Ny)*du_dy
                + nu * lap_u
                - dp_dx / rho
            );

            at(v_star,i,j,Ny) = at(v_old,i,j,Ny) + dt * (
                - at(u_old,i,j,Ny)*dv_dx
                - at(v_old,i,j,Ny)*dv_dy
                + nu * lap_v
            );
        }

        // Pressure RHS = divergence of tentative velocity
        std::fill(rhs.begin(), rhs.end(), 0.0);
        for (int i = 1; i < Nx-1; ++i)
        for (int j = 1; j < Ny-1; ++j)
        {
            if (mask[i*Ny+j]) continue;
            at(rhs,i,j,Ny) = (rho/dt) * (
                (at(u_star,i+1,j,Ny) - at(u_star,i-1,j,Ny)) / (2*dx) +
                (at(v_star,i,j+1,Ny) - at(v_star,i,j-1,Ny)) / (2*dy)
            );
        }

        // Pressure Poisson using Gauss-Seidel
        for (int k = 0; k < POISSON; ++k)
        {
            p_old = p;

            for (int i = 1; i < Nx-1; ++i)
            for (int j = 1; j < Ny-1; ++j)
            {
                if (mask[i*Ny+j]) continue;
                at(p,i,j,Ny) = (
                    (at(p_old,i+1,j,Ny) + at(p_old,i-1,j,Ny)) * dy*dy +
                    (at(p_old,i,j+1,Ny) + at(p_old,i,j-1,Ny)) * dx*dx
                    - at(rhs,i,j,Ny) * dx*dx*dy*dy
                ) / (2*(dx*dx + dy*dy));
            }

            // Pressure BCs
            for (int i = 0; i < Nx; ++i) {
                at(p,i,0,Ny)    = at(p,i,1,Ny);
                at(p,i,Ny-1,Ny) = at(p,i,Ny-2,Ny);
            }
            for (int j = 0; j < Ny; ++j) {
                at(p,0,j,Ny)    = at(p,1,j,Ny);
                at(p,Nx-1,j,Ny) = at(p,Nx-2,j,Ny);
            }
            for (int i = 0; i < Nx; ++i)
            for (int j = 0; j < Ny; ++j)
                if (mask[i*Ny+j]) at(p,i,j,Ny) = 0.0;
        }

        // Velocity correction 
        for (int i = 1; i < Nx-1; ++i)
        for (int j = 1; j < Ny-1; ++j)
        {
            if (mask[i*Ny+j]) continue;
            at(u,i,j,Ny) = at(u_star,i,j,Ny)
                         - dt/rho * (at(p,i+1,j,Ny) - at(p,i-1,j,Ny)) / (2*dx);
            at(v,i,j,Ny) = at(v_star,i,j,Ny)
                         - dt/rho * (at(p,i,j+1,Ny) - at(p,i,j-1,Ny)) / (2*dy);
        }

        // Boundary conditions 

        // Obstacle (no-slip)
        for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            if (mask[i*Ny+j]) { at(u,i,j,Ny)=0; at(v,i,j,Ny)=0; }

        // Inlet: parabolic profile  u = 4*y*(Ly-y)/Ly^2
        for (int j = 0; j < Ny; ++j)
        {
            at(u,0,j,Ny) = 4.0 * y[j] * (Ly - y[j]) / (Ly*Ly);
            at(v,0,j,Ny) = 0.0;
        }

        // Outlet: zero-gradient
        for (int j = 0; j < Ny; ++j)
        {
            at(u,Nx-1,j,Ny) = at(u,Nx-2,j,Ny);
            at(v,Nx-1,j,Ny) = at(v,Nx-2,j,Ny);
        }

        // Top/bottom walls (no-slip)
        for (int i = 0; i < Nx; ++i)
        {
            at(u,i,0,Ny)    = 0; at(v,i,0,Ny)    = 0;
            at(u,i,Ny-1,Ny) = 0; at(v,i,Ny-1,Ny) = 0;
        }

        // Perturbation for vortex generation
        double amp = (n == 0) ? 0.01 : 0.001;
        for (int i = 0; i < Nx; ++i)
        for (int j = 0; j < Ny; ++j)
            at(u,i,j,Ny) += amp * gauss(rng);

        // Adaptive time step
        double max_spd = 1e-8;
        for (int i = 0; i < Nx*Ny; ++i)
            max_spd = std::max(max_spd, std::sqrt(u[i]*u[i] + v[i]*v[i]));
        double dt_visc = dx*dx / nu;
        double dt_conv = dx / max_spd;
        dt = 0.2 * std::min(dt_visc, dt_conv);
    }

    // Write solution to text file
    write_data("pipe_flow_data.txt", u, v, p, x, y, mask, Nx, Ny);
    std::cout << "Simulation complete.\n";
    return 0;
}
