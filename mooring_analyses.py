# Rowan Brown, 13.12.2024
# Munich, Germany

import xarray as xr 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 

def open_mooring_ml_data():
    """Opens CTD data in the mixed layer from the Weddell Sea mooring.
    We're interested because it seems to show a convective plume."""

    # open the .mat file and specifically retrieve the day ('jul'), temp ('T'), salinity ('S') and pressure ('P')
    with open('../filepaths/mooring_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/CTD/Mooring/BGC_SBE.mat'
    mat = spio.loadmat(filepath)['SBE'] # SBE refers to Sea Bird (the instumentation company)
    jul, T, S, P  = mat['jul'][0], mat['T'][0], mat['S'][0], mat['P'][0] # un-nesting the data
    
    # creating list of dates that we'll use to interpolate the data onto
    # specifically, we'll take daily averages of the measurements (perhaps I should weight these...)
    new_day_coord = list(range(85,462)) # we're going to interpolate onto these days
    start_date = datetime(2020,12,31,0,0,0) # basically just assuming this is the start date
    new_datetime_coords = [start_date + timedelta(days=i) for i in new_day_coord] # for the ds

    def daily_avg_mooring_data(inmat): 
        # locally-defined func for getting the average of a variable over one day
        out = np.empty((len(new_day_coord),6)) # initializing a nan array
        out[:] = np.nan
        for i in range(len(jul)):
            if len(inmat[i]) > 0: # basically if the sensor wasn't broken and the data therefore filled with nans
                date = pd.to_datetime([start_date + timedelta(days=i[0]) for i in jul[i]])
                df = pd.DataFrame(data={'date': date, 'var': [i[0] for i in inmat[i]]})
                df = df[(df['date'] > '2021-03-26') & (df['date'] < '2022-04-07')] # masking start+end dates
                new_df = df.groupby(by=df['date'].dt.date).mean()
                out[:,i] = np.array(new_df['var'])
        return out

    # create the xarray dataset
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
    f, ax = plt.subplots(figsize=(3, 2))
    ds.T.sel(depth=[-50,-135,-220]).plot.contourf('day','depth',ax=ax,levels=20,cbar_kwargs={'label': 'Temperature ($\degree C$)'})
    ax.set_ylabel('Depth ($m$)')
    ax.set_xlabel('Date')
    ax.set_title('Temperature at Weddell Sea mooring')
    plt.savefig('Temperature.png',bbox_inches='tight',dpi=450)

# plotting temperature
def sal_hovm(ds):
    f, ax = plt.subplots(figsize=(3, 2))
    ds.S.sel(depth=[-50,-135,-220]).plot.contourf('day','depth',ax=ax,levels=10,cbar_kwargs={'label': 'Salinity ($PSU$)'})
    ax.set_ylabel('Depth ($m$)')
    ax.set_xlabel('Date')
    ax.set_title('Salinity at Weddell Sea mooring')
    plt.savefig('Salinity.png',bbox_inches='tight',dpi=450)

if __name__=="__main__":
    ds = open_mooring_ml_data()
    temp_hovm(ds)
    sal_hovm(ds)
