# Rowan Brown, 13.12.2024
# Munich, Germany

import xarray as xr 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 
import pydap
import mooring_analyses

def get_woa_Weddell_mooring(period:str,decade:int,var):
    """
    Accesses the T and S seasonal climatology data from WOA.
    Saves as .nc in the group WOA directory.

    Parameters:  
        period (str):       month (e.g., "jan", "feb" etc.) or season ('winter', 'spring', 'summer', or 'autumn')
        dataset (int):      2015, 2005, 1995
        var (str):          't', 's'

    Returns datarray of either T or S at the Weddell Sea mooring.
    """
    
    period_dict = {'annual': '00',
                   'jan': '01', 'feb': '02', 'mar': '03', 'apr': '04', 
                   'may': '05', 'jun': '06', 'jul': '07', 'aug': '08', 
                   'sep': '09', 'oct': '10', 'nov': '11', 'dec': '12', 
                   'winter': '13', 'spring': '14', 'summer': '15', 'autumn': '16'}
    decade_dict = {2015: 'B5C2', 2005: 'A5B4', 1995: '95A4'} #note 1995 doesn't have monthly data

    urls = {'t': "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/temperature/netcdf/"+decade_dict[decade]+"/0.25/woa23_"+decade_dict[decade]+"_t"+period_dict[period]+"_04.nc",
            's': "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/salinity/netcdf/"+decade_dict[decade]+"/0.25/woa23_"+decade_dict[decade]+"_s"+period_dict[period]+"_04.nc"}

    # Accessing the data using PyDap
    remote_data = xr.open_dataset(urls[var],decode_times=False,engine='pydap')
    remote_data = remote_data[var+'_an'].sel({'lat': -69, 'lon': -27},method='nearest')
    
    print('Successfully accessed '+var+' WOA data for '+period+', '+str(decade))
    return remote_data

def save_Weddell_mooring_nc():
    """
    Extracts the temp and salinity from the cell nearest the Weddell Sea mooring and saves to an nc.
    """

    months = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    seasons = ['winter', 'spring', 'summer', 'autumn']
    decades = [2015, 2005] #note 1995 doesn't have monthly data AND 2005 has /a lot/ of NANs

    for var in ['t', 's']: 
        for decade in decades:
            da1 = get_woa_Weddell_mooring(months[0],decade,var).assign_attrs(history='WOA data near the Weddell Sea mooring at 69S 27W accessed by Rowan Brown, '+datetime.today().strftime('%Y-%m-%d'))
            for month in months[1:]:
                da2= get_woa_Weddell_mooring(month,decade,var)
                da1 = xr.concat([da1,da2],dim='time')
            with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
            filepath = dirpath + '/WOA_monthly_'+var+'_'+str(decade)+'.nc'             # filepath to the processed data 
            da1.to_netcdf(filepath)
            da1 = get_woa_Weddell_mooring(seasons[0],decade,var).assign_attrs(history='WOA data near the Weddell Sea mooring at 69S 27W accessed by Rowan Brown, '+datetime.today().strftime('%Y-%m-%d'))
            for season in seasons[1:]:
                da2= get_woa_Weddell_mooring(season,decade,var)
                da1 = xr.concat([da1,da2],dim='time')
            with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
            filepath = dirpath + '/WOA_monthly_'+var+'_'+str(decade)+'.nc'             # filepath to the processed data 
            da1.to_netcdf(filepath)
    
def plot_WOA_mooring_t_timeseries():
    """
    Hard-coded function for making plots comparing the WOA temp. data to the mooring temp. data.
    """
    
    # open the mooring data
    ds_mooring = mooring_analyses.open_mooring_ml_data()
    ds_mooring = ds_mooring.resample(day='ME').mean().rename({'day':'month'}) # WOA is monthly so resample
    ds_mooring['month'] = ds_mooring['month'].dt.month # extract the month as an integer 
    ds_mooring = ds_mooring.isel(month=slice(1,13)) # remove the "middle" 12 months to match the climatology 

    # open the WOA data
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    ds_woa_2015 = xr.open_dataset(dirpath + '/WOA_monthly_'+'t'+'_'+str(2015)+'.nc',decode_times=False)
    ds_woa_2015 = ds_woa_2015.rename({'time':'month'}) 
    ds_woa_2015['month'] = ds_woa_2015['month'] - 35.5 # months are saved as arbitrary integers 
    ds_woa_2015['month'] = ds_woa_2015['month'].astype(int) # convert to int 
    ds_woa_2015 = ds_woa_2015.reindex({'month': ds_mooring['month']}) # reindex means instead of 1-12 we have 4-3

    # rename the months as consecutive numbers so that they plot properly
    # hacky but it works
    ds_mooring['month'] = [1,2,3,4,5,6,7,8,9,10,11,12]
    ds_woa_2015['month'] = [1,2,3,4,5,6,7,8,9,10,11,12]

    depths = [50,90,135,170,220,250] # these are the depths we're looking at
    f = plt.figure(figsize=(6, 6)) 
    plt.rcParams["font.family"] = "serif" # change the base font
    gs = f.add_gridspec(len(depths), hspace=0.2) # layout
    axs = gs.subplots(sharex=True)               # layout
    for n,d in enumerate(depths): # plotting in a loop
        ds_mooring['T'].sel(depth=(-1)*d).plot(x='month',ax=axs[n],color='k',label='Mooring (Apr 2021-Mar 2022)')
        ds_woa_2015['t_an'].interp(depth=d).plot(x='month',ax=axs[n],color='k',linestyle='dashed',label='WOA mean (2015-2022)')
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12]) # manually tell it the ticks so that you have one every month 

    # fixing the axs' pretty factor...
    for n,ax in enumerate(axs):
        ax.set_xlim([1,12]) # remove whitespace on edges 
        ax.grid(True) 
        ax.set_xticklabels(['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar'])
        ax.label_outer() # remove all labels within the interior of the plot
        ax.set_ylabel(depths[n],rotation=0,labelpad=15,va="center",size=9) # setting the left (depth) labels
        ax.set_xlabel('') 
        ax.set_title('')
        ax.tick_params(right=True, labelright=True, left=False, labelleft=False,size=9) # moving temps to the right
        ax.tick_params(axis='both',which='both',bottom=False,top=False,size=9)  # removing xticks (they're ugly)
    
    ax.tick_params(axis='both',which='both',bottom=True,top=False,size=9) # formatting the month ticks/labels

    f.suptitle('Temperature in the Weddell Sea at 69$\degree$S, 27$\degree$W\nMooring vs WOA climatology',y=1.00,fontsize=12)
    f.text(0.03, 0.5, 'Depth ($m$)', va='center', rotation='vertical', fontsize=11)
    f.text(0.99, 0.5, 'Temperature ($\degree C$)', va='center', rotation='vertical', fontsize=11)

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.5),
          ncol=2, fancybox=True, fontsize=9)
    
    plt.savefig('Figures/WOA_vs_mooring_temperature_6x6.pdf',format='pdf',bbox_inches='tight')
    plt.savefig('Figures/WOA_vs_mooring_temperature_6x6.png',bbox_inches='tight',dpi=450)

def plot_WOA_mooring_s_timeseries():
    """
    Hard-coded function for making plots comparing the WOA salt data to the mooring salt data.
    """
    
    # open the mooring data
    ds_mooring = mooring_analyses.open_mooring_ml_data()
    ds_mooring = ds_mooring.resample(day='ME').mean().rename({'day':'month'}) # WOA is monthly so resample
    ds_mooring['month'] = ds_mooring['month'].dt.month # extract the month as an integer 
    ds_mooring = ds_mooring.isel(month=slice(1,13)) # remove the "middle" 12 months to match the climatology 

    # open the WOA data
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    ds_woa_2015 = xr.open_dataset(dirpath + '/WOA_monthly_'+'s'+'_'+str(2015)+'.nc',decode_times=False)
    ds_woa_2015 = ds_woa_2015.rename({'time':'month'}) 
    ds_woa_2015['month'] = ds_woa_2015['month'] - 35.5 # months are saved as arbitrary integers 
    ds_woa_2015['month'] = ds_woa_2015['month'].astype(int) # convert to int 
    ds_woa_2015 = ds_woa_2015.reindex({'month': ds_mooring['month']}) # reindex means instead of 1-12 we have 4-3

    # rename the months as consecutive numbers so that they plot properly
    # hacky but it works
    ds_mooring['month'] = [1,2,3,4,5,6,7,8,9,10,11,12]
    ds_woa_2015['month'] = [1,2,3,4,5,6,7,8,9,10,11,12]

    depths = [50,135,220] # these are the depths we're looking at
    f = plt.figure(figsize=(6, 3.5)) 
    plt.rcParams["font.family"] = "serif" # change the base font
    gs = f.add_gridspec(len(depths), hspace=0.2) # layout
    axs = gs.subplots(sharex=True)               # layout
    for n,d in enumerate(depths): # plotting in a loop
        ds_mooring['S'].sel(depth=(-1)*d).plot(x='month',ax=axs[n],color='k',label='Mooring (Apr 2021-Mar 2022)')
        ds_woa_2015['s_an'].interp(depth=d).plot(x='month',ax=axs[n],color='k',linestyle='dashed',label='WOA mean (2015-2022)')
    plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12]) # manually tell it the ticks so that you have one every month 

    # fixing the axs' pretty factor...
    for n,ax in enumerate(axs):
        ax.set_xlim([1,12]) # remove whitespace on edges 
        ax.grid(True) 
        ax.set_xticklabels(['Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec','Jan','Feb','Mar'])
        ax.label_outer() # remove all labels within the interior of the plot
        ax.set_ylabel(depths[n],rotation=0,labelpad=15,va="center",size=9) # setting the left (depth) labels
        ax.set_xlabel('') 
        ax.set_title('')
        ax.tick_params(right=True, labelright=True, left=False, labelleft=False,size=9) # moving temps to the right
        ax.tick_params(axis='both',which='both',bottom=False,top=False,size=9)  # removing xticks (they're ugly)
    
    ax.tick_params(axis='both',which='both',bottom=True,top=False,size=9) # formatting the month ticks/labels

    f.suptitle('Salinity in the Weddell Sea at 69$\degree$S, 27$\degree$W\nMooring vs WOA climatology',y=1.09,fontsize=12)
    f.text(0.03, 0.5, 'Depth ($m$)', va='center', rotation='vertical', fontsize=11)
    f.text(1.01, 0.5, 'Salinity ($PSU$)', va='center', rotation='vertical', fontsize=11)

    axs[0].legend(loc='upper center', bbox_to_anchor=(0.5, 1.42),
          ncol=2, fancybox=True, fontsize=9)
    
    plt.savefig('Figures/WOA_vs_mooring_salinity_6x35.pdf',format='pdf',bbox_inches='tight')
    plt.savefig('Figures/WOA_vs_mooring_salinity_6x35.png',bbox_inches='tight',dpi=450)

if __name__=="__main__":
    #save_Weddell_mooring_nc()
    #plot_WOA_mooring_t_timeseries()
    plot_WOA_mooring_s_timeseries()
    
    