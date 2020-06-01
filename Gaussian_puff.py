import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
import statsmodels.api as sm
import scipy.io as sio
import scipy.stats as stats
import scipy.special as special


# %%
def orthoregress(x, y):
    '''
    Orthogonal regression.
    Parameters
    ----------
    x: np.array
    y: np.array

    Returns
    -------
    (slope, intercpet): (float, float)
    '''
    linreg = linregress(x, y)
    mod = Model(f)
    dat = Data(x, y, wd=1. / (np.var(x) + 1e-8), we=1. / (np.var(y) + 1e-8))
    # small value is added to var to prevent zero division error
    od = ODR(dat, mod, beta0=linreg[0:2])
    out = od.run()
    return list(out.beta)


def f(p, x):
    """Basic linear regression 'model' for use with ODR"""
    return (p[0] * x) + p[1]


# %%
def speed_behind_vehicle(U, h, x_p, z_p, y_p, x_v, y_v):
    '''
    Calculate wake speed behind the vehicle.

    Parameters
    ----------
    U: float,
        Vehicle speed (m/s).
    h: float,
        Vehicle height (m/s).
    x_p: float
        Puff x (m).
    z_p: float
        Puff z (m).
    y_p: float
        Puff y (m).
    x_v: float
        Vehicle x (m).
    y_v: float
        Vehicle y (m).
    z_v: float
        Vehicle z (m).

    Returns
    -------
    u_w: float
        Puff center speed caused by vehicle movement (m/s).
    '''
    s = (x_v - x_p)
    s[s < 0] = 0
    h_w = h * (s / h) ** 0.25
    xi = z_p / h_w
    eta = (y_p - y_v) / h_w
    u_w = U * (s / h) ** (-0.75) * xi * np.exp(-(xi ** 2. + eta ** 2.) / 8.)
    u_w[np.isnan(u_w)] = 0
    return u_w


def speed_canyon(u_H, v_H, H, B, z_p, y_p, z0):
    '''
    Calculate speed due to ambient wind.
    Parameters
    ----------
    u_H: float
        Ambient wind speed along the canyon (m/s).
    v_H: float
        Ambient wind speed cross the canyon (m/s).
    H: float
        Height of the buildings (m).
    B: float
        Width of the canyon (m).
    z_p:
        Height of the puff (m).
    z0:
        Roughness height (m).

    Returns
    -------
    (u, v, w): (float, float, float)
        Puff center wind speed due to ambiend wind (m/s).
    '''
    k = np.pi / B
    gamma = np.exp(-2 * k * H)
    z1 = z_p - H

    return (u_H * np.log((z_p + z0) / z0) / np.log((z0 + H) / z0),
            v_H * (np.exp(k * z1) * (1 + k * z1) - gamma * np.exp(-k * z1) * (
                    1 - k * z1)) * np.sin(k * y_p) / (1 - gamma),
            -v_H * k * z1 * (np.exp(k * z1) - gamma * np.exp(-k * z1)) * np.cos(
                k * y_p / (1 - gamma))
            )


def cal_sigma(sigma_x0, sigma_y0, sigma_z0, alpha1, alpha2, u, v, w, dt):
    '''
    Calculate dispersion coefficients.

    Parameters
    ----------
    sigma_x0: float
        x-directional dispersion.
    sigma_y0: float
        y-directional dispersion.
    sigma_z0: float
        z-directional dispersion
    alpha1: float
        Parameter as in Eq. (12).
    alpha2: float
        Parameter as in Eq. (12).
    u: float
        x-directional wind speed (m/s).
    v: float
        y-directional wind speed (m/s).
    w: float
        z-directional wind speed (m/s).
    dt: float
        Time step.

    Returns
    -------
    sigma_x, sigma_y, sigma_z: (float, float, float)
        Updated dispersion coefficients.
    '''
    tur_u = alpha1 * np.sqrt(
        (alpha2 * u ** 2. + (1 - alpha2) * (v ** 2. + w ** 2.)))
    tur_v = alpha1 * np.sqrt(
        (alpha2 * v ** 2. + (1 - alpha2) * (u ** 2. + w ** 2.)))
    tur_w = alpha1 * np.sqrt(
        (alpha2 * w ** 2. + (1 - alpha2) * (u ** 2. + v ** 2.)))
    return (sigma_x0 + tur_u * dt,
            sigma_y0 + tur_v * dt,
            sigma_z0 + tur_w * dt)


def gaussian_puff(x, y, z, xp, yp, zp, sigmax, sigmay, sigmaz, B):
    X = np.exp(-0.5 * (x - xp) ** 2. / sigmax ** 2.) / np.sqrt(
        np.pi * 2) / sigmax
    Y = (np.exp(-0.5 * (y - yp) ** 2. / sigmay ** 2.) + np.exp(
        -0.5 * (y + yp) ** 2. / sigmay ** 2.) + np.exp(
        -0.5 * (2 * B - y - yp) ** 2. / sigmay ** 2.)) / np.sqrt(
        np.pi * 2) / sigmay
    Z = (np.exp(-0.5 * (z - zp) ** 2. / sigmaz ** 2.) + np.exp(
        -0.5 * (z + zp) ** 2. / sigmaz ** 2.)) / np.sqrt(
        np.pi * 2) / sigmaz
    return X * Y * Z


# %% Define canyon and aerodynamic properties
L = 600.0  # Canyon length (m)
B = 20.  # Canyon width (m)
H = 20.  # Building height (m)
l = [9, 3, 3, 3, 3, 3]  # Vehicle length (m)
b = [2.5, 1.8, 1.8, 1.8, 1.8, 1.8]  # Vehicle width (m)
h = [2.5, 1.67, 1.5, 1.5, 1.5, 1.5]  # Vehicle height (m)
u_H = 0  # Along-canyon wind speed (m/s)
v_H = -3.0  # Cross-canyon wind speed (m/s)
z0 = 1.0  # Roughness length (m)
dt = 0.05  # Puff and model resolution (s)
alpha_1 = 0.25  # Dispersion parameter
alpha_2 = 0.1  # Dispersion parameter
n = 1500  # Iteration number, 300 initial steps for

# %% Define acceleration and initial speed for vehicles
# Define NGV path
a1 = np.r_[
    np.zeros(300), np.linspace(0, 1, 50), np.linspace(1, 0, 50),\
    np.linspace(0,1,50), np.linspace(1, 0, 50), np.zeros(103),\
    np.linspace(0, -2.4, 125), np.linspace(-2.4, 0, 125), np.zeros(57),\
    np.linspace(0, 3, 5), np.linspace(3, 0.5, 5), np.linspace(0.5, 3,50),\
    np.linspace(3, 0, 50), np.zeros(15), np.linspace(0, 3, 20),\
    np.linspace(3, 0, 20), np.zeros(75), np.linspace(0, 1, 75),\
    np.linspace(1, 0, 75), np.linspace(0, -3, 100), np.linspace(-3, 0, 100)]
u1 = np.array([np.trapz(a1[:i], dx=dt) for i in range(len(a1))]) + 10
u1[:300] = 0
x1 = np.array([np.trapz(u1[:i], dx=dt) for i in range(len(u1))])
a_v1 = np.array([0] * 300 + [-0.03125] * 160 + [0.03125] * 160 + [0] * 460 + [
                    0.03125] * 160 + [-0.03125] * 160 + [0] * 100)
v1 = np.array([np.trapz(a_v1[:i], dx=dt) for i in range(len(a_v1))])
y1 = np.array([np.trapz(v1[:i], dx=dt) for i in range(len(a_v1))]) + 13.5
# Define mobile lab path
a2 = np.r_[
    np.zeros(300), np.zeros(50), np.linspace(0, 1.5, 125),\
    np.linspace(1.5, 0, 125), np.zeros(50), np.linspace(-0, -1, 40),\
    np.linspace(-1, -2.8, 90), np.linspace(-2.8, 0, 90), np.zeros(40),\
    np.linspace(0, 3, 30), np.linspace(3, 0, 30), np.zeros(50),\
    np.linspace(0, 1.75, 100), np.linspace(1.75, 0, 100), np.zeros(
        50), np.linspace(0, 2, 40), np.linspace(2, 0, 40), np.zeros(
        50), np.linspace(0, 2, 25), np.linspace(
        2, 0, 25), np.zeros(50)]
u2 = np.array([np.trapz(a2[:i], dx=dt) for i in range(len(a2))]) + 6.5
u2[:300] = 0
a_v2 = np.zeros(1500)
v2 = np.array([np.trapz(a_v2[:i], dx=dt) for i in range(len(a_v2))])
y2 = np.array([np.trapz(v2[:i], dx=dt) for i in range(len(a_v2))]) + 11.5
x2 = np.array([np.trapz(u2[:i], dx=dt) for i in range(len(u2))])
# Define cross-lane vehicle path
a3 = np.zeros(1500)
u3 = np.ones(1500) * -18.0
x3 = 550 + np.array([np.trapz(u3[:i], dx=dt) for i in range(len(u3))])
y3 = 8.5 * np.ones(1500)
a4 = np.zeros(1500)
u4 = np.ones(1500) * -18.0
u4[0:400] = 0
y4 = 8.5 * np.ones(1500)
x4 = 600 + np.array([np.trapz(u4[:i], dx=dt) for i in range(len(u3))])
a5 = np.zeros(1500)
u5 = np.ones(1500) * -18.0
u5[0:900] = 0
y5 = 8.5 * np.ones(1500)
x5 = 600 + np.array([np.trapz(u5[:i], dx=dt) for i in range(len(u3))])
# Define in-lane vehicle path
a6 = np.r_[
    np.zeros(985), np.linspace(0, 1, 25), np.linspace(1, 0, 25),\
    np.linspace(0,-1,25), np.linspace(-1, 0, 25), np.zeros(415)]
a_v6 = np.r_[
    np.zeros(985), np.linspace(0, 1, 25), np.linspace(1, 0, 25),\
    np.linspace(0, -1, 25), np.linspace(-1, 0, 25), np.zeros(415)]

u6 = np.array([np.trapz(a6[:i], dx=dt) for i in range(len(a6))])
x6 = np.array([np.trapz(u6[:i], dx=dt) for i in range(len(u1))]) + 310
v6 = np.array([np.trapz(a_v6[:i], dx=dt) for i in range(len(a_v1))])
y6 = np.array([np.trapz(v6[:i], dx=dt) for i in range(len(a_v1))]) + 13.7

# %% convert arrays to matrix
a_v = np.c_[a1, a2, a3, a4, a5, a6]
u_v = np.c_[u1, u2, u3, u4, u5, u6]
v_v = np.c_[
    np.zeros(1500), v2, np.zeros(1500), np.zeros(1500), np.zeros(1500), v6]
x_v = np.c_[x1, x2, x3, x4, x5, x6]
y_v = np.c_[y1, y2, y3, y4, y5, y6]

# %% Define puff matrices, axis = 0 is time, axis=1 is puff number
u_p = np.zeros((1500, 1500, 6))
v_p = np.zeros((1500, 1500, 6))
w_p = np.zeros((1500, 1500, 6))
sigma_x = np.zeros((1500, 1500, 6))
sigma_y = np.zeros((1500, 1500, 6))
sigma_z = np.zeros((1500, 1500, 6))
x_p = np.zeros((1500, 1500, 6))
y_p = np.zeros((1500, 1500, 6))
z_p = np.zeros((1500, 1500, 6))


# %% Simulate puff movements
t_start = [300, 300, 0, 400, 900, 900]
for i in range(1, 1499):
    for k in range(6):
        if i >= t_start[k]:
            # Add new puffs
            if k in [0, 1]:
                x_p[i, i, k] = x_v[i, k] - l[k]
                y_p[i, i, k] = y_v[i, k]
            elif k == 5:
                x_p[i, i, k] = x_v[i, k] - 0.71 * 1.5
                y_p[i, i, k] = y_v[i, k] - 0.71 * 1.5
            else:
                x_p[i, i, k] = x_v[i, k] + l[k] / 2.
                y_p[i, i, k] = y_v[i, k]

            # Update center locations of previously emitted puffs

            x_p[i, :i, k] = np.array(
                [np.trapz(u_p[:i, j, k], dx=dt) + x_p[j, j, k] for j in
                 range(i)])
            y_p[i, :i, k] = np.array(
                [np.trapz(v_p[:i, j, k], dx=dt) + y_p[j, j, k] for j in
                 range(i)])
            z_p[i, :i, k] = np.array(
                [np.trapz(w_p[:i, j, k], dx=dt) + z_p[j, j, k] for j in
                 range(i)])

            # Update wake related speed
            # 0 and 1 are NGVs and mobile lab
            # 2-4 are cross-lane vehicles, speed and distance are opposite
            # 5 are in-lane but moves 45deg
            if (u_v[i, k] ** 2. + v_v[i, k] ** 2.) > 0:
                if k in [0, 1]:
                    tmp_speed_w = speed_behind_vehicle(
                        np.sqrt(u_v[i, k] ** 2. + v_v[i, k] ** 2.), h[k],
                        x_p[i, :i + 1, k] + l[k] / 2,
                        z_p[i, :i + 1, k],
                        y_p[i, :i + 1, k],
                        x_v[i, k], y_v[i, k])
                    tmp_u_w = tmp_speed_w * u_v[i, k] / np.sqrt(
                        u_v[i, k] ** 2. + v_v[i, k] ** 2.)
                    tmp_v_w = tmp_speed_w * v_v[i, k] / np.sqrt(
                        u_v[i, k] ** 2. + v_v[i, k] ** 2.)
                elif k == 5:
                    tmp_speed_w = speed_behind_vehicle(
                        np.sqrt(u_v[i, k] ** 2. + v_v[i, k] ** 2.), h[k],
                        x_p[i, :i + 1, k] + l[k] / 2 * 0.71,
                        z_p[i, :i + 1, k],
                        y_p[i, :i + 1, k] + l[k] / 2 * 0.71,
                        x_v[i, k], y_v[i, k])
                    tmp_u_w = tmp_speed_w * u_v[i, k] / np.sqrt(
                        u_v[i, k] ** 2. + v_v[i, k] ** 2.)
                    tmp_v_w = tmp_speed_w * v_v[i, k] / np.sqrt(
                        u_v[i, k] ** 2. + v_v[i, k] ** 2.)
                else:
                    tmp_speed_w = speed_behind_vehicle(
                        np.sqrt(u_v[i, k] ** 2. + v_v[i, k] ** 2.), h[k],
                        -x_p[i, :i + 1, k],
                        z_p[i, :i + 1, k],
                        y_p[i, :i + 1, k],
                        -x_v[i, k], y_v[i, k])
                    tmp_u_w = tmp_speed_w * u_v[i, k] / np.sqrt(
                        u_v[i, k] ** 2. + v_v[i, k] ** 2.)
                    tmp_v_w = tmp_speed_w * v_v[i, k] / np.sqrt(
                        u_v[i, k] ** 2. + v_v[i, k] ** 2.)
            else:
                tmp_u_w = 0
                tmp_v_w = 0

            # Update above canyon speed
            tmp_u, tmp_v, tmp_w = speed_canyon(u_H, v_H, H, B,
                                               z_p[i, :i + 1, k],
                                               y_p[i, :i + 1, k],
                                               z0)
            # Update puff speed
            u_p[i, :i + 1, k] = tmp_u + tmp_u_w
            v_p[i, :i + 1, k] = tmp_v + tmp_v_w
            w_p[i, :i + 1, k] = tmp_w

            #  Initial height of puff
            z_p[i, i, k] = 0.3
            sigma_x[i, i, k] = 0.25 * (h[k] + b[k])
            sigma_y[i, i, k] = 0.5 * b[k]
            sigma_z[i, i, k] = 0.5 * h[k]
            # Update new sigma
            sigma_x[i + 1, :i + 1, k], sigma_y[i + 1, :i + 1, k], sigma_z[i + 1,
                                                                  :i + 1,
                                                                  k] = cal_sigma(
                sigma_x[i, :i + 1, k], sigma_y[i, :i + 1, k],
                sigma_z[i, :i + 1, k],
                alpha_1,
                alpha_2,
                u_p[i, :i + 1, k], v_p[i, :i + 1, k], w_p[i, :i + 1, k], dt)

# %% Define emission properties
q = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2]
q_idle = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2]
t_start = [300, 300, 0, 400, 900, 900]

# %% Calculate speed and accelartion dependent emissions
tmp_a = a_v.copy()
tmp_a[tmp_a < 0] = 0
speed_emission_coeff = 1 * u_v + 3 * tmp_a
speed_emission_coeff[:, 1:6] = 1 * np.abs(u_v[:, 1:6]) + 3 * tmp_a[:, 1:6]
speed_emission_coeff[speed_emission_coeff <= 0] = 0
speed_emission_coeff2 =  np.abs(u_v) + 1.5 * tmp_a
speed_emission_coeff2[:, 1:6] = 1 * np.abs(u_v[:, 1:6]) + 1.5 * tmp_a[:, 1:6]
speed_emission_coeff2[speed_emission_coeff <= 0] = 0
# %% Calculate smooth Gaussian puff model
c = np.zeros((1499, 6))
for t in range(1499):
    for k in range(6):
        tmp_c = 0
        for i in range(1, t):
            if i > t_start[k]:
                tmp_c += 150 * (
                        0.3 * q_idle[k] + q[k] * speed_emission_coeff[
                    i, k]) * gaussian_puff(x_v[t, 1], y_v[t, 1], 2.0,
                                           x_p[t, i, k], y_p[t, i, k],
                                           z_p[t, i, k],
                                           sigma_x[t, i, k],
                                           sigma_y[t, i,
                                                   k],
                                           sigma_z[t, i, k],
                                           B)
        c[t, k] = tmp_c

c2 = np.array([])
for t in range(1499):
    tmp_c = 0
    for i in range(1, t):
        for k in range(1):
            if i > t_start[k]:
                tmp_c += 5 * q[k] * (
                        0.5 + speed_emission_coeff2[
                    i, k]) * gaussian_puff(x_v[t, 1], y_v[t, 1], 2.0,
                                           x_p[t, i, k],
                                           y_p[t, i, k],

                                           z_p[t, i, k],

                                           sigma_x[t, i, k],
                                           sigma_y[t, i, k],
                                           sigma_z[t, i, k],
                                           B)
    c2 = np.r_[c2, tmp_c]

# %% Perform plume chasing method
co2_new = c[300:1499:2, :]
ch4_new = c2[300:1499:2]
slope = np.zeros(580)
slope2 = np.zeros(580)
offset = np.zeros(580)
pvalue = np.zeros(580)

tvalue = np.zeros(580)
rsqr = np.zeros(580)
linreg = []
# fig, axes = plt.subplots(4, 5, sharex=True, sharey=True)
half_window = 10
for t in range(half_window, 600 - half_window):
    # tmp_co2 = np.sum(co2_new[t:t + 20,[0,2,3,4,5]], axis=1)
    tmp_co2 = np.nansum(co2_new[t:t + 2 * half_window, :], axis=1)
    # tmp_co2 = co2_new[t:t+20]
    tmp_ch4 = ch4_new[t:t + 2 * half_window]
    X = sm.add_constant(tmp_co2)
    Y = tmp_ch4
    fit_r = sm.OLS(Y, X).fit()
    slope[t - half_window], offset[t - half_window] = orthoregress(tmp_co2,
                                                                   tmp_ch4)
    slope2[t - half_window] = (np.nanmean(tmp_ch4 / tmp_co2))

    rsqr[t - half_window] = np.corrcoef(tmp_co2, tmp_ch4)[0, 1] ** 2.
    # axes[(t-200)/5, (t-220)%5].plot(tmp_co2, tmp_ch4, '.')
    pvalue[t - half_window] = linregress(tmp_co2, tmp_ch4)[3]
    # axes[(t-140)/5, (t-140)%5].plot(tmp_co2, tmp_ch4, '.')

    # linreg.append(tmp_lin)

# %% Remove periods impacted by other vehicles
idx = (rsqr > 0.5) & (np.nansum(co2_new[half_window:600 - half_window, :],
                                axis=1) > 5) & (
              ch4_new[half_window:600 - half_window] > 0.1) & (
              u_v[(300 + half_window * 2):(1500 - half_window * 2):2, 1] > (
              5. / 3.6))
new_idx = idx.copy()
for i in range(0, 600 - 2 * half_window):
    if (rsqr[i] < 0.5):
        # print i
        new_idx[(i - half_window):(i + 1 * half_window)] = 0

# %% Plot results
fig, axes = plt.subplots(4, 1, figsize=(6.5, 4.5), sharex=True)
labels = ['NGV', 'mobile lab', 'V1', 'V2', 'V3', 'V4']
for i in range(0, 6):
    axes[0].plot(co2_new[:, i] + 400, label=labels[i])
axes[0].plot(np.sum(co2_new, axis=1)+400, 'k', label='Sum')
axes[0].set_ylim([400, 800])
axes[0].legend(frameon=False, fontsize=6, ncol=2, loc=2)
axes[1].plot(ch4_new + 2)
axes[1].set_ylim([2, 15])
axes[2].plot(range(10, 590), slope, c='tab:blue', label='Orthogonal regression')
axes[2].plot(range(10, 590), slope2, c='tab:orange', label='Quotient')
tmp_slope = slope.copy()
tmp_slope2 = slope2.copy()
tmp_slope[idx == 0] = np.nan
tmp_slope2[idx == 0] = np.nan
axes[2].plot((ch4_new / co2_new[:, 0]),
             c='tab:green', label='True value')
axes[2].legend(frameon=False, fontsize=6)
axes[2].set_ylim([0, 0.1])
axes[3].plot(range(10, 590), rsqr)
axes[0].set_ylabel('CO$_2$ (ppmv)', fontsize=9)
axes[1].set_ylabel('CH$_4$ (ppmv)', fontsize=9)
axes[2].set_ylabel('$\Delta$CH$_4$/$\Delta$CO$_2$\n(ppmv/ppmv)', fontsize=9)
axes[3].set_ylabel('R$^2$', fontsize=9)
axes[3].set_xticks([150, 300, 450, 600])
axes[3].set_xticklabels(['15', '30', '45', '60'])
axes[3].set_ylim([0, 1])
axes[3].set_xlabel('t (sec)')
axes[3].set_xlim([0, 600])
[plt.setp(axes[i].get_xticklabels(), fontsize=9) for i in range(4)]
[plt.setp(axes[i].get_yticklabels(), fontsize=9) for i in range(4)]
[axes[i].fill_between(range(10, 590), new_idx * 1000., color='r', alpha=0.2,
                      lw=0) for i in range(4)]
plt.tight_layout()
plt.savefig('smooth_puff.svg', dpi=600, format='svg')

# %% Random walk Gaussian puff model, calculate puff center
x_p_r = np.zeros((1500, 1500, 6))
y_p_r = np.zeros((1500, 1500, 6))
z_p_r = np.zeros((1500, 1500, 6))
for k in range(6):
    for i in range(1, 1500):
        if (i) >= t_start[k]:
            n = i - t_start[k]
            p = np.array([[special.erfinv(
                2. * np.random.uniform(0, 1.0) - 1) for ii in range(n)] for
                kk in range(3)])
            dx = u_p[i, t_start[k]:i, k] * dt + \
                 np.sqrt((sigma_x[i, t_start[k]:i, k] ** 2. - sigma_x[(i - 1),
                                                              t_start[k]:i,
                                                              k] ** 2.)) * p[0,
                                                                           :]
            dy = v_p[i, t_start[k]:i, k] * dt + \
                 np.sqrt((sigma_y[i, t_start[k]:i, k] ** 2. - sigma_y[(i - 1),
                                                              t_start[k]:i,
                                                              k] ** 2.)) * p[1,
                                                                           :]
            dz = w_p[i, t_start[k]:i, k] * dt + \
                 np.sqrt((sigma_z[i, t_start[k]:i, k] ** 2. - sigma_z[(i - 1),
                                                              t_start[k]:i,
                                                              k] ** 2.)) * p[2,
                                                                           :]
            x_p_r[i, t_start[k]:i, k] = x_p_r[
                                        i - 1,
                                        t_start[k]:(
                                            i),
                                        k] + dx
            y_p_r[i, t_start[k]:(i), k] = y_p_r[
                                          i - 1,
                                          t_start[k]:(
                                              i),
                                          k] + dy
            z_p_r[i, t_start[k]:(i), k] = z_p_r[
                                          i - 1,
                                          t_start[k]:(
                                              i),
                                          k] + dz
            x_p_r[i, (i), k] = x_p[
                i, i, k]
            y_p_r[i, (i), k] = y_p[
                i, i, k]
            z_p_r[i, (i), k] = z_p[
                i, i, k]

y_p_r = np.abs(y_p_r)
y_p_r[y_p_r > 20] = 40 - y_p_r[y_p_r > 20]
z_p_r = np.abs(z_p_r)

# %% Calculate concentrations
c = np.zeros((1499, 6))
for t in range(1499):
    for k in range(6):
        tmp_c = 0
        for i in range(1, t):
            if i > t_start[k]:
                tmp_c += 150 * (
                        0.2 * q_idle[k] + q[k] * speed_emission_coeff[
                    i, k]) * gaussian_puff(x_v[t, 1], y_v[t, 1], 2.0,
                                           x_p_r[t, i, k], y_p_r[t, i, k],
                                           z_p_r[t, i, k],

                                           sigma_x[t, i, k] / np.sqrt(2.),
                                           sigma_y[t, i,
                                                   k] / np.sqrt(2.),
                                           sigma_z[t, i, k] / np.sqrt(2.),
                                           B)
        c[t, k] = tmp_c

c2 = np.array([])
for t in range(1499):
    tmp_c = 0
    for i in range(1, t):
        for k in range(1):
            if i > t_start[k]:
                tmp_c += 5 * q[k] * (
                        2 + speed_emission_coeff2[
                    i, k]) * gaussian_puff(x_v[t, 1], y_v[t, 1], 2.0,
                                           x_p_r[t, i, k],
                                           y_p_r[t, i, k],
                                           z_p_r[t, i, k],
                                           sigma_x[t, i, k] / np.sqrt(2.),
                                           sigma_y[t, i, k] / np.sqrt(2.),
                                           sigma_z[t, i, k] / np.sqrt(2.),
                                           B)
    c2 = np.r_[c2, tmp_c]

# %% Perform plume chasing
co2_new = (c[300:1499:2, :] + c[299:1500:2, :]) / 2
ch4_new = (c2[300:1499:2] + c2[299:1500:2]) / 2
half_window = 10
slope = np.zeros(600 - half_window * 2)
slope2 = np.zeros(600 - half_window * 2)
slope3 = np.zeros(600 - half_window * 2)
offset = np.zeros(600 - half_window * 2)
pvalue = np.zeros(600 - half_window * 2)

tvalue = np.zeros(600 - 2 * half_window)
rsqr = np.zeros(600 - 2 * half_window)
linreg = []
for t in range(half_window, 600 - half_window):
    tmp_co2 = np.nansum(co2_new[t:t + 2 * half_window, :], axis=1)
    tmp_ch4 = ch4_new[t:t + 2 * half_window]
    X = sm.add_constant(tmp_co2)
    Y = tmp_ch4
    fit_r = sm.OLS(Y, X).fit()
    slope[t - half_window], offset[t - half_window] = orthoregress(tmp_co2,
                                                                   tmp_ch4)
    slope2[t - half_window] = (np.nanmean(tmp_ch4 / tmp_co2))
    slope3[t - half_window] = linregress(tmp_co2, tmp_ch4)[0]

    rsqr[t - half_window] = np.corrcoef(tmp_co2, tmp_ch4)[0, 1] ** 2.

    pvalue[t - half_window] = linregress(tmp_co2, tmp_ch4)[3]

    # linreg.append(tmp_lin)

# %%
idx = (rsqr > 0.5) & (np.nansum(co2_new[half_window:600 - half_window, :],
                                axis=1) > 5) & (
              ch4_new[half_window:600 - half_window] > 0.1) & (
              u_v[(300 + half_window * 2):(1500 - half_window * 2):2, 1] > (
              8. / 3.6))
new_idx = idx.copy()
for i in range(0, 600 - 2 * half_window):
    if (rsqr[i] < 0.5):
        new_idx[(i - half_window):(i + 1 * half_window)] = 0

# %% Plot results
fig, axes = plt.subplots(4, 1, figsize=(6.5, 4.5), sharex=True)
labels = ['NGV', 'mobile lab', 'V1', 'V2', 'V3', 'V4']
for i in range(0, 6):
    axes[0].plot(co2_new[:, i] + 400, label=labels[i])
axes[0].plot(np.sum(co2_new, axis=1)+400, 'k', label='Sum')
axes[0].set_ylim([400, 800])
axes[0].legend(frameon=False, fontsize=6, ncol=2, loc=2)
axes[1].plot(ch4_new + 2)
axes[1].set_ylim([2, 15])
axes[2].plot(range(10, 590), slope, c='tab:blue', label='Orthogonal regression')
axes[2].plot(range(10, 590), slope2, c='tab:orange', label='Quotient')
tmp_slope = slope.copy()
tmp_slope2 = slope2.copy()
tmp_slope[idx == 0] = np.nan
tmp_slope2[idx == 0] = np.nan
axes[2].plot((ch4_new / co2_new[:, 0]),
             c='tab:green', label='True value')
axes[2].legend(frameon=False, fontsize=6)
# axes[2].set_ylim(0.000,0.07)
axes[2].set_ylim([0, 0.2])
axes[3].plot(range(10, 590), rsqr)
axes[0].set_ylabel('CO$_2$ (ppmv)', fontsize=9)
axes[1].set_ylabel('CH$_4$ (ppmv)', fontsize=9)
axes[2].set_ylabel('$\Delta$CH$_4$/$\Delta$CO$_2$\n(ppmv/ppmv)', fontsize=9)
axes[3].set_ylabel('R$^2$', fontsize=9)
axes[3].set_xticks([150, 300, 450, 600])
axes[3].set_xticklabels(['15', '30', '45', '60'])
axes[3].set_ylim([0, 1])
axes[3].set_xlabel('t (sec)')
axes[3].set_xlim([0, 600])
[plt.setp(axes[i].get_xticklabels(), fontsize=9) for i in range(4)]
[plt.setp(axes[i].get_yticklabels(), fontsize=9) for i in range(4)]
[axes[i].fill_between(range(10, 590), new_idx * 1000., color='r', alpha=0.2,
                      lw=0) for i in range(4)]
plt.tight_layout()
plt.savefig('random_puff.svg', dpi=600, format='svg')

# %% Check profile from RMGUFFER
test_x = []
test_y = []
test_z = []

tmp_x_p_r = np.zeros((1500, 100))
tmp_y_p_r = np.zeros((1500, 100))
tmp_z_p_r = np.zeros((1500, 100))
for kk in range(100):
    x_p_r = np.zeros(1500)
    y_p_r = np.zeros(1500)
    z_p_r = np.zeros(1500)
    for k in range(1):
        for i in range(400, 1500):
            if (i) >= t_start[k]:
                p = np.array([special.erfinv(
                    2. * np.random.uniform(0, 1.0) - 1) for
                    kkk in range(3)])
                dx = u_p[i, 400, k] * dt + \
                     np.sqrt(
                         (sigma_x[i, 400, k] ** 2. - sigma_x[(i - 1),
                                                             400,
                                                             k] ** 2.)) * p[0]
                dy = v_p[i, 400, k] * dt + \
                     np.sqrt(
                         (sigma_y[i, 400, k] ** 2. - sigma_y[(i - 1),
                                                             400,
                                                             k] ** 2.)) * p[1]
                dz = w_p[i, 400, k] * dt + \
                     np.sqrt(
                         (sigma_z[i, 400, k] ** 2. - sigma_z[(i - 1),
                                                             400,
                                                             k] ** 2.)) * p[2]
                x_p_r[i] = x_p_r[i - 1] + dx
                y_p_r[i] = y_p_r[i - 1] + dy
                z_p_r[i] = z_p_r[i - 1] + dz
                x_p_r[400] = x_p[
                    400, 400, 0]
                y_p_r[400] = y_p[400, 400, 0]
                z_p_r[400] = z_p[400, 400, 0]

    tmp_x_p_r[:, kk] = x_p_r
    tmp_y_p_r[:, kk] = y_p_r
    tmp_z_p_r[:, kk] = z_p_r

tmp_y_p_r = np.abs(tmp_y_p_r)
tmp_y_p_r[tmp_y_p_r > 20] = 40 - tmp_y_p_r[tmp_y_p_r > 20]
tmp_z_p_r = np.abs(tmp_z_p_r)
# %% Plot results
plt.close('all')
fig, axes = plt.subplots(3, 2, figsize=(6.5, 5))
axes[0, 0].plot(tmp_x_p_r[300:], c='tab:blue', alpha=0.3)
axes[0, 0].plot(x_p[300:1499, 400, 0], c='k')
axes[1, 0].plot(tmp_y_p_r[300:], c='tab:blue', alpha=0.3)
axes[1, 0].plot(y_p[300:1499, 400, 0], c='k')
axes[2, 0].plot(tmp_z_p_r[300:, :-1], c='tab:blue', alpha=0.3)
axes[2, 0].plot(tmp_z_p_r[300:, -1], c='tab:blue', alpha=0.3, label='Center of the puff\n(random walk Puffer)')
axes[2, 0].plot(z_p[300:1499, 400, 0], c='k', label='Center of the puff (PUFFER)')
axes[2, 0].legend(frameon=False, fontsize=7)
ylabels = ['x (m)', 'y (m)', 'z (m)']
for i in range(3):
    axes[i, 0].set_xlim([0, 1200])
    axes[i, 0].set_xticks([0, 300, 600, 900, 1200])
    axes[i, 0].set_xticklabels(['0', '15', '30', '45', '60'])
    axes[i, 0].set_xlabel('t (sec)', fontsize=9)
    axes[i, 0].set_ylabel(ylabels[i], fontsize=9)

x_p_grid = np.linspace(30, 70, 100)
y_p_grid = np.linspace(0, 20, 100)
z_p_grid = np.linspace(0, 20, 100)
sigmax = sigma_x[900,400,0]/np.sqrt(2)
sigmay = sigma_x[900,400,0]/np.sqrt(2)
sigmaz = sigma_x[900,400,0]/np.sqrt(2)
tmp_x_dis = np.zeros((len(x_p_grid), 50))
tmp_y_dis = np.zeros((len(x_p_grid), 50))
tmp_z_dis = np.zeros((len(x_p_grid), 50))

for k in range(50):
    xp = tmp_x_p_r[900, k]
    yp = tmp_y_p_r[900, k]
    zp = tmp_z_p_r[900, k]
    tmp_x_dis[:,k] = np.exp(-0.5 * (x_p_grid - xp) ** 2. / sigmax ** 2.) / np.sqrt(
        np.pi * 2) / sigmax
    tmp_y_dis[:,k] = (np.exp(-0.5 * (y_p_grid - yp) ** 2. / sigmay ** 2.) + np.exp(
        -0.5 * (y_p_grid + yp) ** 2. / sigmay ** 2.) + np.exp(
        -0.5 * (2 * B - y_p_grid  - yp) ** 2. / sigmay ** 2.)) / np.sqrt(
        np.pi * 2) / sigmay
    tmp_z_dis[:,k] = (np.exp(-0.5 * (z_p_grid - zp) ** 2. / sigmaz ** 2.) + np.exp(
        -0.5 * (z_p_grid + zp) ** 2. / sigmaz ** 2.)) / np.sqrt(
        np.pi * 2) / sigmaz

xp = x_p[900, 400, 0]
yp = y_p[900, 400, 0]
zp = z_p[900, 400, 0]
sigmax = sigma_x[900,400,0]
sigmay = sigma_y[900,400,0]
sigmaz = sigma_z[900,400,0]
true_x_dis = np.exp(-0.5 * (x_p_grid - xp) ** 2. / sigmax ** 2.) / np.sqrt(
        np.pi * 2) / sigmax
true_y_dis = (np.exp(-0.5 * (y_p_grid - yp) ** 2. / sigmay ** 2.) + np.exp(
    -0.5 * (y_p_grid + yp) ** 2. / sigmay ** 2.) + np.exp(
    -0.5 * (2 * B - y_p_grid - yp) ** 2. / sigmay ** 2.)) / np.sqrt(
    np.pi * 2) / sigmay
true_z_dis = (np.exp(-0.5 * (z_p_grid - zp) ** 2. / sigmaz ** 2.) + np.exp(
    -0.5 * (z_p_grid + zp) ** 2. / sigmaz ** 2.)) / np.sqrt(
    np.pi * 2) / sigmaz
[axes[0, 1].plot(x_p_grid, tmp_x_dis[:,k], c='tab:blue', alpha=0.3) for k in range(50)]
axes[0, 1].plot(x_p_grid, true_x_dis, 'k')
axes[0, 1].plot(x_p_grid, np.mean(tmp_x_dis, axis=1), 'r')
axes[0, 1].set_xlim([30, 70])
axes[0, 1].set_xlabel('x (m)', fontsize=9)
[axes[1, 1].plot(y_p_grid, tmp_y_dis[:,k], c='tab:blue', alpha=0.3) for k in range(50)]
axes[1, 1].plot(y_p_grid, true_y_dis, 'k')
axes[1, 1].plot(y_p_grid, np.mean(tmp_y_dis, axis=1), 'r')
axes[1, 1].set_xlim([0, 20])
axes[1, 1].set_xlabel('y (m)', fontsize=9)

[axes[2, 1].plot(z_p_grid, tmp_z_dis[:,k], c='tab:blue', alpha=0.3) for k in range(49)]
axes[2, 1].plot(z_p_grid, tmp_z_dis[:,49], c='tab:blue', alpha=0.3, label='Puff profile (random walk PUFFER)')
axes[2, 1].plot(z_p_grid, true_z_dis, 'k', label='Puff profile (PUFFER)')
axes[2, 1].plot(z_p_grid, np.mean(tmp_z_dis, axis=1), 'r', label='Ensemble mean')
axes[2,1].legend(frameon=False, fontsize=7)
axes[2, 1].set_xlim([0, 20])
axes[2, 1].set_xlabel('z (m)', fontsize=9)
[axes[i, 1].set_ylabel('Normalized\nconcentrations', fontsize=9) for i in range(3)]
[[plt.setp(axes[i,j].get_xticklabels(), fontsize=9) for i in range(3)] for j in range(2)]
[[plt.setp(axes[i,j].get_yticklabels(), fontsize=9) for i in range(3)] for j in range(2)]
plt.tight_layout(h_pad=0.05)
plt.savefig('RWPUFFER_ensemble_mean.svg', dpi=600, format='svg')
# %% Repeat random walk PUFFER for 50 times to compare results
new_slope = np.zeros(50)
new_quo = np.zeros(50)
old_slope = np.zeros(50)
old_quo = np.zeros(50)
for kkk in range(50):
    test_x = []
    test_y = []

    test_z = []
    x_p_r = np.zeros((1500, 1500, 6))
    y_p_r = np.zeros((1500, 1500, 6))
    z_p_r = np.zeros((1500, 1500, 6))
    for k in range(6):
        for i in range(1, 1500):
            if (i) >= t_start[k]:
                n = i - t_start[k]
                p = np.array([[special.erfinv(
                    2. * np.random.uniform(0, 1.0) - 1) for ii in range(n)] for
                    kk in range(3)])
                dx = u_p[i, t_start[k]:i, k] * dt + \
                     np.sqrt(
                         (sigma_x[i, t_start[k]:i, k] ** 2. - sigma_x[(i - 1),
                                                              t_start[k]:i,
                                                              k] ** 2.)) * p[0,
                                                                           :]
                dy = v_p[i, t_start[k]:i, k] * dt + \
                     np.sqrt(
                         (sigma_y[i, t_start[k]:i, k] ** 2. - sigma_y[(i - 1),
                                                              t_start[k]:i,
                                                              k] ** 2.)) * p[1,
                                                                           :]
                dz = w_p[i, t_start[k]:i, k] * dt + \
                     np.sqrt(
                         (sigma_z[i, t_start[k]:i, k] ** 2. - sigma_z[(i - 1),
                                                              t_start[k]:i,
                                                              k] ** 2.)) * p[2,
                                                                           :]
                x_p_r[i, t_start[k]:i, k] = x_p_r[
                                            i - 1,
                                            t_start[k]:(
                                                i),
                                            k] + dx
                y_p_r[i, t_start[k]:(i), k] = y_p_r[
                                              i - 1,
                                              t_start[k]:(
                                                  i),
                                              k] + dy
                z_p_r[i, t_start[k]:(i), k] = z_p_r[
                                              i - 1,
                                              t_start[k]:(
                                                  i),
                                              k] + dz
                x_p_r[i, (i), k] = x_p[
                    i, i, k]
                y_p_r[i, (i), k] = y_p[
                    i, i, k]
                z_p_r[i, (i), k] = z_p[
                    i, i, k]

    y_p_r = np.abs(y_p_r)
    y_p_r[y_p_r > 20] = 40 - y_p_r[y_p_r > 20]
    z_p_r = np.abs(z_p_r)

    # %%
    c = np.zeros((1499, 6))
    for t in range(1499):
        for k in range(6):
            tmp_c = 0
            for i in range(1, t):
                if i > t_start[k]:
                    tmp_c += 150 * (
                            0.2 * q_idle[k] + q[k] * speed_emission_coeff[
                        i, k]) * gaussian_puff(x_v[t, 1], y_v[t, 1], 2.0,
                                               x_p_r[t, i, k], y_p_r[t, i, k],
                                               z_p_r[t, i, k],

                                               sigma_x[t, i, k] / np.sqrt(2.),
                                               sigma_y[t, i,
                                                       k] / np.sqrt(2.),
                                               sigma_z[t, i, k] / np.sqrt(2.),
                                               B)
            c[t, k] = tmp_c

    c2 = np.array([])
    for t in range(1499):
        tmp_c = 0
        for i in range(1, t):
            for k in range(1):
                if i > t_start[k]:
                    tmp_c += 5 * q[k] * (
                            0.5 + speed_emission_coeff2[
                        i, k]) * gaussian_puff(x_v[t, 1], y_v[t, 1], 2.0,
                                               x_p_r[t, i, k],
                                               y_p_r[t, i, k],
                                               z_p_r[t, i, k],
                                               sigma_x[t, i, k] / np.sqrt(2.),
                                               sigma_y[t, i, k] / np.sqrt(2.),
                                               sigma_z[t, i, k] / np.sqrt(2.),
                                               B)
        c2 = np.r_[c2, tmp_c]

    # %%
    co2_new = (c[300:1499:2, :] + c[299:1500:2, :]) / 2
    ch4_new = (c2[300:1499:2] + c2[299:1500:2]) / 2
    half_window = 10
    slope = np.zeros(600 - half_window * 2)
    slope2 = np.zeros(600 - half_window * 2)
    slope3 = np.zeros(600 - half_window * 2)
    offset = np.zeros(600 - half_window * 2)
    pvalue = np.zeros(600 - half_window * 2)

    # %%
    tvalue = np.zeros(600 - 2 * half_window)
    rsqr = np.zeros(600 - 2 * half_window)
    linreg = []
    for t in range(half_window, 600 - half_window):
        tmp_co2 = np.nansum(co2_new[t:t + 2 * half_window, :], axis=1)
        tmp_ch4 = ch4_new[t:t + 2 * half_window]
        X = sm.add_constant(tmp_co2)
        Y = tmp_ch4
        fit_r = sm.OLS(Y, X).fit()
        slope[t - half_window], offset[t - half_window] = orthoregress(tmp_co2,
                                                                       tmp_ch4)
        slope2[t - half_window] = (np.nanmean(tmp_ch4 / tmp_co2))
        slope3[t - half_window] = linregress(tmp_co2, tmp_ch4)[0]

        rsqr[t - half_window] = np.corrcoef(tmp_co2, tmp_ch4)[0, 1] ** 2.

        pvalue[t - half_window] = linregress(tmp_co2, tmp_ch4)[3]


    # %%
    idx = (rsqr > 0.5) & (np.nansum(co2_new[half_window:600 - half_window, :],
                                    axis=1) > 5) & (
                  ch4_new[half_window:600 - half_window] > 0.1) & (
                  u_v[(300 + half_window * 2):(1500 - half_window * 2):2, 1] > (
                  5. / 3.6))
    new_idx = idx.copy()
    for i in range(0, 600 - 2 * half_window):
        if (rsqr[i] < 0.5):
            new_idx[(i - half_window):(i + 1 * half_window)] = 0

    # %%
    new_slope[kkk] = np.nanmean(slope[new_idx])
    new_quo[kkk] = np.nanmean(slope2[new_idx])
    old_slope[kkk] = np.nanmean(slope[idx])
    old_quo[kkk] = np.nanmean(slope2[idx])

# %% Print out results and simulated ER
gprw_results = pd.DataFrame({'new_slope': new_slope,
                             'new_quo': new_quo,
                             'old_slope': old_slope,
                             'old_quo': old_quo})
gprw_results.to_csv('GaussianRandoWalk_results.csv')
