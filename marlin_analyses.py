# Rowan Brown, 13.01.2025
# Munich, Germany

import xarray as xr 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 

def open_marlin_data():
    """Opens Marlin float data in the mixed layer.
    We're interested because it seems to show a convective plume."""

    # getting part of the file path, which is saved in a text file to avoid publishing it to GitHub
    # the [0] accesses the first line, and the [:-1] removes the newline tag
    with open('../filepaths/marlin_filepath') as f: dirpath = f.readlines()[0][:-1] 

    # creating the full filepaths to the .nc file
    fp = dirpath + '/uptempo_all.nc' 
    
    # opening the .nc files
    ds = xr.open_dataset(fp)
    ds['time'] = [datetime(1, 1, 1, 0, 0) + timedelta(days=time - 367) for time in ds['time'].values] # 376 is necessary because you can only start datetime at 01-01-0001, plus an extra 2 days from who-knows-where (but I tested this with MATLAB datevec and 367 is correct)
    ds = ds.where((ds.temperature<2) & (ds.temperature>-2),np.nan)
    return ds

    # dealing with dates...
    new_day_coord = list(range(85,462)) # we're going to interpolate onto these days, measure from the start of the year (I think)
    start_date = datetime(2020,12,31,0,0,0) # I'm basically just assuming this is the start date
    new_datetime_coords = [start_date + timedelta(days=i) for i in new_day_coord] # list need for creating the dataset

    # locally-defined function for getting the time-weighted average of a variable over one day
    def daily_avg_mooring_data(inmat): 
        
        # initializing a nan array, size is [number of days] x [number of depths]
        out = np.empty((len(new_day_coord),6)) 
        out[:] = np.nan
        
        # looping through the measured timesteps
        for i in range(len(jul)):
            if len(inmat[i]) > 0: # basically if we have a non-emtpy array
                date = pd.to_datetime([start_date + timedelta(days=i[0]) for i in jul[i]]) # creating datetimes for each entry
                df = pd.DataFrame(data={'date': date, 'var': [i[0] for i in inmat[i]]}) # filling a pandas dataframe
                df['weights'] = [i.total_seconds() for i in df.diff().date] # seconds elapsed between measurements
                df = df.set_index('date') 
                df = df[(df.index > '2021-03-26') & (df.index < '2022-04-07')] # truncating start and end dates 
                # taking the time-weighted average every 24 hours 
                # weights don't align perfectly with each day, but it's close enough
                new_df = df.resample('D').apply(lambda df : np.sum(df['weights']*df['var'])/np.sum(df['weights']))
                out[:,i] = np.array(new_df) # writing to the "out" array

        return out
    
    # creating an xarray dataset
    ds = xr.Dataset(
        data_vars=dict(
            T=(["day","depth"], daily_avg_mooring_data(T)),
            S=(["day","depth"], daily_avg_mooring_data(S)),
            P=(["day","depth"], daily_avg_mooring_data(P)),
        ),
        coords=dict(
            day=new_datetime_coords,
            depth=[-50,-90,-135,-170,-220,-250],
        ),
        attrs=dict(description="Mooring data..."),
    )

    return ds

# plotting temperature
def temp_hovm(ds):
    """Created a Hovmöller plot of temperature."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, axs = plt.subplots(4,2,figsize=(8, 6),sharex=True, sharey=True)
    for id,ax in enumerate(f.axes):
        float_id = str(ds['buoy'].astype(int).values[id])
        cax = ds['temperature'].isel(buoy=id).plot.contourf('time','depth',ax=ax,levels=10,add_colorbar=False)
        cbar = f.colorbar(cax, ax=ax)
        cbar.ax.tick_params(labelsize=9)
        ax.grid(True) 
        ax.set_ylabel('',fontsize=9)
        ax.set_xlabel('',fontsize=9)
        ax.set_title('Float ID: '+float_id,fontsize=11)
        ax.tick_params(axis='both',labelsize=9)
        ax.invert_yaxis()
    for ax in f.get_axes():
        ax.label_outer()
    f.suptitle('Temperature from Marlin Floats',fontsize=12)
    f.supylabel('Depth ($m$)')
    f.tight_layout()
    plt.savefig('Figures/Marlin_temperature_hovm_8x6.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Marlin_temperature_hovm_4x3_' + float_id[id] + '.pdf',format='pdf',bbox_inches='tight')

if __name__=="__main__":
    ds = open_marlin_data()
    temp_hovm(ds)
