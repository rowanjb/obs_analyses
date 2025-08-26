# For checking out the Copernicus 0.125 deg-res atmospheric/wind data from
# https://data.marine.copernicus.eu/product/WIND_GLO_PHY_L4_MY_012_006/download

import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime as dt


def wind():
    with open('../filepaths/copernicus') as f:
        dir_fp = f.readlines()[0][:-1]
    file = ('cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H_multi-vars_'
            '27.94W-26.06W_69.94S-68.06S_2021-04-01-2022-04-01.nc')
    file_fp = dir_fp + file

    ds = xr.open_dataset(file_fp)

    eastward_wind = ds['eastward_wind'].interp(longitude=-27.0048,
                                               latitude=-69.0005)
    northward_wind = ds['northward_wind'].interp(longitude=-27.0048,
                                                 latitude=-69.0005)

    wind = (eastward_wind**2 + northward_wind**2)**0.5

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    wind.rolling(time=24, center=True).mean().plot(ax=ax)
    ax.set_ylabel('Wind speed ($m$ $s^{-1}$)')
    ax.set_title('10-m wind speed at mooring\n(69.0005S, 27.0048W)',
                 fontsize=12)
    ax.text(0.01,
            0.01,
            ('Source: Copernicus Global Ocean Hourly Reprocessed Sea '
             'Surface\nWind and Stress from Scatterometer and Model'
             '(0.125°'+r'$\times$'+'0.125° resolution)'),
            verticalalignment='bottom',
            horizontalalignment='left',
            transform=ax.transAxes,
            color='black',
            fontsize=7)
    ax.vlines(x=dt.strptime('2021-09-06', '%Y-%m-%d'),
              ymin=0,
              ymax=25,
              colors='red')
    ax.text(dt.strptime('2021-09-06', '%Y-%m-%d'),
            19.9,
            'September 6',
            verticalalignment='bottom',
            horizontalalignment='left',
            va='top',
            color='red',
            fontsize=7)
    ax.set_ylim(0, 20)
    ax.set_xlim(dt.strptime('2021-04-01', '%Y-%m-%d'),
                dt.strptime('2022-04-01', '%Y-%m-%d'))
    ax.set_xlabel('')
    plt.grid()
    plt.savefig('Figures/mooring_atm_figs/wind_cmems.png', dpi=600)


def wind_ERA5():
    with open('../filepaths/ERA5') as f:
        dir_fp = f.readlines()[0][:-1]
    file = '/ERA5_mooring/ERA5_mooring_u10_v10.nc'
    file_fp = dir_fp + file

    ds = xr.open_dataset(file_fp)

    eastward_wind = ds['u10'].interp(longitude=-27.0048, latitude=-69.0005)
    northward_wind = ds['v10'].interp(longitude=-27.0048, latitude=-69.0005)

    wind = (eastward_wind**2 + northward_wind**2)**0.5

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    wind.rolling(valid_time=24, center=True).mean().plot(ax=ax)
    ax.set_ylabel('Wind speed ($m$ $s^{-1}$)')
    ax.set_title('10-m wind speed at mooring\n(69.0005S, 27.0048W)',
                 fontsize=12)
    ax.text(0.01, 0.01, 'Source: ERA5 (0.25°'+r'$\times$'+'0.25° resolution)',
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='black', fontsize=7)
    ax.vlines(x=dt.strptime('2021-09-06', '%Y-%m-%d'),
              ymin=0,
              ymax=25,
              colors='red')
    ax.text(dt.strptime('2021-09-06', '%Y-%m-%d'),
            19.9,
            'September 6',
            verticalalignment='bottom',
            horizontalalignment='left',
            va='top',
            color='red',
            fontsize=7)
    ax.set_ylim(0, 20)
    ax.set_xlim(dt.strptime('2021-04-01', '%Y-%m-%d'),
                dt.strptime('2022-04-01', '%Y-%m-%d'))
    ax.set_xlabel('')
    plt.grid()
    plt.subplots_adjust(top=0.815, right=0.96, left=0.14)
    plt.savefig('Figures/mooring_atm_figs/wind_ERA5.png', dpi=600)


def temp():
    with open('../filepaths/ERA5') as f:
        dir_fp = f.readlines()[0][:-1]
    file = '/ERA5_mooring/ERA5_mooring_t2m_sp.nc'
    file_fp = dir_fp + file

    ds = xr.open_dataset(file_fp)

    t2m = ds['t2m'].interp(longitude=-27.0048, latitude=-69.0005) - 273.15

    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(5, 2.5))
    t2m.rolling(valid_time=24, center=True).mean().plot(ax=ax) 
    ax.set_ylabel('Air temperature ($℃$)')
    ax.set_title('2-m air temperature at mooring\n(69.0005S, 27.0048W)',
                 fontsize=12)
    ax.text(0.01, 0.01, 'Source: ERA5 (0.25°'+r'$\times$'+'0.25° resolution)',
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes, color='black', fontsize=7)
    ax.vlines(x=dt.strptime('2021-09-06','%Y-%m-%d'),
              ymin=-35,
              ymax=2,
              colors='red')
    ax.text(dt.strptime('2021-09-06', '%Y-%m-%d'),
            1.8,
            'September 6',
            verticalalignment='bottom',
            horizontalalignment='left',
            va='top',
            color='red',
            fontsize=7)
    ax.set_ylim(-35, 2)
    ax.set_xlim(dt.strptime('2021-04-01', '%Y-%m-%d'),
                dt.strptime('2022-04-01', '%Y-%m-%d'))
    ax.set_xlabel('')
    plt.grid()
    plt.subplots_adjust(top=0.815, right=0.96, left=0.14)
    plt.savefig('Figures/mooring_atm_figs/temp_ERA5.png', dpi=600)


if __name__ == "__main__":
    temp()
    wind()
    wind_ERA5()
