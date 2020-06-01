'''
Calculate ER using mobile lab data and identify plume data.
The code is created to run with Python 2.7.
Following packages are needed.

Numpy
Scipy
Pandas

Lase edited by Da Pan,
02/16/2020.
Email: dp7@princeton.edu or pd.phy.pku@gmail.com
'''
import numpy as np
import pandas as pd
from scipy.odr import Model, Data, ODR
from scipy.stats import linregress
import scipy.io as sio

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


# %% Section 1. Calculate ERs using 10 Hz data
# ERs are already included in the data files.
# This section can be skipped unless the time window is changed.
date_list = ['0610', '0611', '0612']  # Date string list
half_window = 10  # Half time window for ratio and R2 calculation
for date in date_list[0:3]:
    print date
    # Read raw 10 Hz data.
    data = pd.read_csv(date + '_10Hz_with_ratio_1s_final.csv', parse_dates=['TS']).set_index(
        'TS')
    # Determine the baseline concentration using moving 2nd percentile.
    data_3min_2nd = data.resample('3min').agg(lambda x: x.quantile(0.02))
    data['CH4bsn'] = data_3min_2nd['TSCH4']
    data['CH4bsn'] = data['CH4bsn'].interpolate()
    data['CO2bsn'] = data_3min_2nd['TSCO2']
    data['CO2bsn'] = data['CO2bsn'].interpolate()
    data['NH3bsn'] = data_3min_2nd['TSNH3']
    data['NH3bsn'] = data['NH3bsn'].interpolate()
    # Calculate peaks due to local emissions
    data['CH4peak'] = data['TSCH4'] - data['CH4bsn']
    data['CO2peak'] = data['TSCO2'] - data['CO2bsn']
    data['NH3peak'] = data['TSNH3'] - data['NH3bsn']
    # Calculate enhancement ratio
    data['ratio'] = np.nan
    data['coeff'] = np.nan
    data['ratio_nh3'] = np.nan
    data['coeff_nh3'] = np.nan
    for i in range(half_window, len(data) - half_window + 1):
        # Truncate CH4 and CO2 data for i-half_window:i+window
        tmp_data = data.iloc[i - half_window:i + half_window].dropna(
            subset=['CH4peak', 'CO2peak'], how='any')
        # Calculate CH4 ER if there are enough data
        if len(tmp_data) > 1:
            a, b = orthoregress(tmp_data['CO2peak'], tmp_data['CH4peak'])
            data['ratio'].iloc[i] = a
            data['coeff'].iloc[i] = \
                (np.corrcoef(tmp_data['CH4peak'], tmp_data['CO2peak']) ** 2)[
                    0, 1]
        # Truncate NH3 and CO2 data for i-half_window:i+window
        tmp_data = data.iloc[i - half_window:i + half_window].dropna(
            subset=['NH3peak', 'CO2peak'], how='any')
        # Calculate NH3 ER if there are enough data
        if len(tmp_data) > 1:
            a, b = orthoregress(tmp_data['CO2peak'], tmp_data['NH3peak'])
            data['ratio_nh3'].iloc[i] = a
            data['coeff_nh3'].iloc[i] = \
                (np.corrcoef(tmp_data['NH3peak'], tmp_data['CO2peak']) ** 2)[
                    0, 1]
        # Print out progress and NH3 and CH4 ERs
        print i, '/', len(data), data['ratio_nh3'].iloc[i], data['ratio'].iloc[
            i]

    data.to_csv(date + '_10Hz_with_ratio_1s_final.csv')

# %% Section 2. Identify plumes and calculate mean ER
date_str = ['0610', '0611', '0612']
thold_list = [0, 0.2, 0.4]  # CH4 threshold for determination of uncertainty
# Tolerance of start and end of the periods that the mobile lab were following NG buses
tol_dt_bus = pd.to_timedelta(20, 's')
# Tolerance of start and end of the periods that the mobile lab were following NG taxis
tor_dt_taxi = pd.to_timedelta(20, 's')
mean_er_bus = np.array([])
mean_er_bus_nh3 = np.array([])
mean_er_taxi = np.array([])
mean_er_taxi_nh3 = np.array([])
mean_er_all = np.array([])
size_er_bus = np.array([])
size_er_taxi = np.array([])
size_er_bus_nh3 = np.array([])
size_er_taxi_nh3 = np.array([])
size_er_all = np.array([])
rhold = 0.5  # Threshold of Rsquared.
er_ch4_bus = np.array([])
er_nh3_bus = np.array([])
er_ch4_taxi = np.array([])
er_nh3_taxi = np.array([])
er_nh3 = np.array([])
# Loop thru dates and thresholds
for date in date_str:
    for thold in thold_list:
        print date, thold
        # Read data with ERs
        data = pd.read_csv(date + '_10Hz_with_ratio_1s_final.csv',
                           parse_dates=['TS']).set_index('TS')
        # Read start and end times of NGV encounters.
        # Although the field is named "bus", it actually contains data for taxis
        # as well.
        t_intv = sio.loadmat("Bus_%s.mat" % (date))['Bus']
        # Convert mat data that contains hour, minute, and second since 00:00
        # of the day to pd.datetime. These time stamps are start time of the
        # encounter. The time stamps from the dash camera were lagged behind
        # the main data logger by 8.1414 hour.
        st_time = pd.to_datetime('2014' + date) + \
                  pd.to_timedelta(t_intv[:, 0], 'h') + \
                  pd.to_timedelta(t_intv[:, 1], 'm') + \
                  pd.to_timedelta(t_intv[:, 2], 's') + \
                  pd.to_timedelta(8.1414, 'h')

        # The 5th column contains flags indicating whether the row is for bus
        # (1) or taxi (0).
        st_bus = st_time[t_intv[:, 5] == 1]
        st_taxi = st_time[t_intv[:, 5] == 0]
        # The 4th column contains durations of the encounter.
        dt_bus = pd.to_timedelta(t_intv[t_intv[:, 5] == 1, 4], 's')
        dt_taxi = pd.to_timedelta(t_intv[t_intv[:, 5] == 0, 4], 's')
        data['flag_bus'] = False
        data['flag_taxi'] = False
        # Mark periods that the mobile lab was following NG buses.
        for st, dt in zip(st_bus, dt_bus):
            data.loc[st - tol_dt_bus:st + dt + tol_dt_bus, 'flag_bus'] = True
        # Mark periods that the mobile lab was following NG taxis.
        for st, dt in zip(st_taxi, dt_taxi):
            data.loc[st - tor_dt_taxi:st + dt + tor_dt_taxi, 'flag_taxi'] = True
        # Apply filters for ER calculation.
        data['valid_bus'] = (data['flag_bus']) & (data['CH4peak'] > thold) & (
                data['CO2peak'] > 10) & (
                                    (data['coeff']) > rhold) & (
                                    data['ratio'] > 0)
        # Apply filters for NH3 ER calculation
        data['valid_bus_nh3'] = (data['flag_bus']) & (
                data['CH4peak'] > thold) & (
                                        data['CO2peak'] > 10) & (
                                        (data['coeff_nh3']) > rhold) & (
                                        data['ratio_nh3'] > 0)
        # Apply filters for ER calculation.
        data['valid_taxi'] = (data['flag_taxi']) & (data['CH4peak'] > thold) & (
                data['CO2peak'] > 10) & (
                                     (data['coeff']) > rhold) & (
                                     data['ratio'] > 0)
        # Apply filters for NH3 ER calculation.
        data['valid_taxi_nh3'] = (data['flag_taxi']) & (
                data['CH4peak'] > thold)  & (
                                         data['CO2peak'] > 10) & (
                                         (data['coeff_nh3']) > rhold)  & (
                                         data['ratio_nh3'] > 0)
        # Apply filters for all ER.
        data['valid_nh3'] = ( data['CO2peak'] > 10) & (
                                         (data['coeff_nh3']) > rhold)& (
                                         data['ratio_nh3'] > 0)
        # Calculate ER for taxi
        mean_er_taxi = np.r_[
            mean_er_taxi, np.mean(data[data['valid_taxi']]['ratio'])]
        mean_er_taxi_nh3 = np.r_[
            mean_er_taxi_nh3, np.mean(
                data[data['valid_taxi_nh3']]['ratio_nh3'])]
        # Calculate ER for bus
        mean_er_bus = np.r_[
            mean_er_bus, np.mean(data[data['valid_bus']]['ratio'])]
        mean_er_bus_nh3 = np.r_[
            mean_er_bus_nh3, np.mean(data[data['valid_bus_nh3']]['ratio_nh3'])]
        mean_er_all = np.r_[mean_er_all, np.mean(data.loc[data['valid_nh3'],'ratio_nh3'])]
        # Size distribution
        size_er_bus = np.r_[size_er_bus, np.sum(data['valid_bus'])]
        size_er_bus_nh3 = np.r_[size_er_bus_nh3, np.sum(data['valid_bus_nh3'])]
        # Size distribution
        size_er_taxi = np.r_[size_er_taxi, np.sum(data['valid_taxi'])]
        size_er_taxi_nh3 = np.r_[size_er_taxi_nh3, np.sum(data['valid_taxi_nh3'])]
        size_er_all = np.r_[size_er_all, np.sum(data['valid_nh3'])]
        # Save er distribution for percentile calculation
        if thold==0.2:
            er_ch4_bus = np.r_[er_ch4_bus, data.loc[data['valid_bus'], 'ratio']]
            er_ch4_taxi = np.r_[er_ch4_taxi, data.loc[data['valid_taxi'], 'ratio']]
            er_nh3_bus = np.r_[er_nh3_bus, data.loc[data['valid_bus_nh3'], 'ratio_nh3']]
            er_nh3_taxi = np.r_[er_nh3_taxi, data.loc[data['valid_taxi_nh3'], 'ratio_nh3']]
            er_nh3 = np.r_[er_nh3, data.loc[data['valid_nh3'], 'ratio_nh3']]

# %% Section 3. Calculate mean ER and uncertainty.
# Calculate plume identification uncertainty
daily_uncertainty_bus = np.array([(mean_er_bus[2] - mean_er_bus[0]) / 2,
                                  (mean_er_bus[5] - mean_er_bus[3]) / 2,
                                  (mean_er_bus[8] - mean_er_bus[6]) / 2])
daily_uncertainty_taxi = np.array([(mean_er_taxi[2] - mean_er_taxi[0]) / 2,
                                   (mean_er_taxi[5] - mean_er_taxi[3]) / 2,
                                   (mean_er_taxi[8] - mean_er_taxi[6]) / 2])
# Get daily mean ER
daily_er_bus = np.array([mean_er_bus[[1, 4, 7]]])
daily_er_taxi = np.array([mean_er_taxi[[1, 4, 7]]])
# Get number of samples for each day
daily_n_bus = np.array([size_er_bus[1], size_er_bus[4], size_er_bus[7]])
daily_n_taxi = np.array([size_er_taxi[1], size_er_taxi[4], size_er_taxi[7]])
# Combine plume identification uncertainty and daily variation.
uncertainty_bus = np.sqrt((np.sum(
    daily_uncertainty_bus ** 2 * daily_n_bus) + np.sum(
    (daily_er_bus - np.mean(daily_er_bus)) ** 2 * daily_n_bus)) / np.sum(
    daily_n_bus))
uncertainty_taxi = np.sqrt((np.sum(
    daily_uncertainty_taxi ** 2 * daily_n_taxi) + np.sum(
    (daily_er_taxi - np.mean(daily_er_taxi)) ** 2 * daily_n_taxi)) / np.sum(
    daily_n_taxi))
# Calculate number of sample weighted mean
mean_taxi = (mean_er_taxi[1] * daily_n_taxi[0] + mean_er_taxi[4] * daily_n_taxi[
    1] + mean_er_taxi[7] * daily_n_taxi[2]) / np.sum(daily_n_taxi)
mean_bus = (mean_er_bus[1] * daily_n_bus[0] + mean_er_bus[4] * daily_n_bus[
    1] + mean_er_bus[7] * daily_n_bus[2]) / np.sum(daily_n_bus)

# %%
print 'Mean CH4:CO2 enhancement ratio for buses: ', mean_bus, ', +/-SE:', uncertainty_bus
print 'Mean CH4:CO2 enhancement ratio for taxis: ', mean_taxi, ', +/-SE:', uncertainty_taxi