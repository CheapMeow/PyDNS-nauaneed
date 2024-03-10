def PyDNS():
    import os
    print(os.getcwd())

    import sys
    sys.path.append(os.getcwd())

    import numpy as np
    from matplotlib import pyplot
    from src import projection_method

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

    ##cylinder
    R = 0.15
    cx = 1.85
    cy = 4

    ##physical variables
    Re = 100
    rho = 1
    nu = rho * 1 * R * 2 / Re
    F = 0
    dt = 1.8e-3

    # boundary conditions
    bc = {'x': 'neumann', 'y': 'free-slip'}

    # initial conditions
    u = np.ones((ny, nx))
    utemp = np.zeros((ny, 3))

    v = np.zeros((ny, nx))
    vtemp = np.zeros((ny, 3))

    if bc['y']=='no-slip':
        u[0, :] = 0
        u[-1, :] = 0
        v[0, :] = 0
        v[-1, :] = 0

    p = np.zeros((ny, nx))
    ptemp = np.zeros((ny, 3))
    dpdx = np.zeros((ny, nx))
    dpdy = np.zeros((ny, nx))
    epsilon = np.zeros((ny, nx))
    uRHS_conv_diff_p = np.zeros((ny, nx))
    uRHS_conv_diff_pp = np.zeros((ny, nx))
    vRHS_conv_diff_p = np.zeros((ny, nx))
    vRHS_conv_diff_pp = np.zeros((ny, nx))

    # ibm
    r = ((xx - cx) ** 2 + (yy - cy) ** 2) ** 0.5
    theta = np.arctan2(yy - cy, xx - cx)

    for i in range(nx):
        for j in range(ny):
            if r[j, i] <= R:
                epsilon[j, i] = 1

    ## pressure_poisson
    nx_sp = nx
    ny_sp = ny

    kx = np.array([(2 * np.pi * i / lx) for i in range(0, (int(nx_sp / 2) - 1))])
    kx = np.append(kx, np.array([(2 * np.pi * (nx_sp - i) / lx) for i in range(int(nx_sp / 2) - 1, nx_sp)]))
    ky = np.array([(np.pi * (i + 1) / ly) for i in range(0, ny_sp)])
    KX, KY = np.meshgrid(kx, ky)
    K = KX ** 2 + KY ** 2

    for stepcount in range(0, nt + 1):
        print("Step=%06i time=%4.6f" % (stepcount, stepcount * dt))

        if (np.mod(stepcount, saveth_iter) == 0) and (stepcount > save_start):
            print("snapshot= %i" % (stepcount / saveth_iter))
            time = stepcount * dt
            index = int(stepcount / saveth_iter)
            np.savetxt("./data/p_t=%.4f_%d.csv" % (time, index), np.asarray(p), delimiter=",")
            np.savetxt("./data/u_t=%.4f_%d.csv" % (time, index), np.asarray(u), delimiter=",")
            np.savetxt("./data/v_t=%.4f_%d.csv" % (time, index), np.asarray(v), delimiter=",")

        ustar, vstar, uRHS_conv_diff, vRHS_conv_diff = projection_method.step1(u, v, nx, ny, nu, x, y, xx, yy, dx, dy,
                                                                               dt, epsilon, F, R, theta, r,
                                                                               uRHS_conv_diff_p, uRHS_conv_diff_pp,
                                                                               vRHS_conv_diff_p, vRHS_conv_diff_pp,
                                                                               dpdx, dpdy, bc)

        # Step2
        ustarstar, vstarstar = projection_method.step2(ustar, vstar, dpdx, dpdy, dt)

        # Step3
        p = projection_method.step3(ustarstar, vstarstar, rho, epsilon, dx, dy, nx_sp, ny_sp, K, dt, bc)

        # Step4
        u, v, dpdx, dpdy = projection_method.step4(ustarstar, vstarstar, p, dx, dy, dt, bc)

        if bc['y'] == 'free-slip':
            u[0, :] = u[1, :].copy()
            u[-1, :] = u[-2, :].copy()
            v[0, :] = 0
            v[-1, :] = 0

        ip_mdot = dy * ((u[0, 0] + u[-1, 0]) / 2 + sum(u[1:-1, 0]))
        op_mdot = dy * ((u[0, -1] + u[-1, -1]) / 2 + sum(u[1:-1, -1]))
        print("mass flow rate ip op diff: %f %f %e" % (ip_mdot, op_mdot, op_mdot-ip_mdot))


        uRHS_conv_diff_pp = uRHS_conv_diff_p.copy()
        vRHS_conv_diff_pp = vRHS_conv_diff_p.copy()

        uRHS_conv_diff_p = uRHS_conv_diff.copy()
        vRHS_conv_diff_p = vRHS_conv_diff.copy()

if __name__ == "__main__":
    import os

    if not os.path.exists('data'):
        os.makedirs('data')

    PyDNS()
