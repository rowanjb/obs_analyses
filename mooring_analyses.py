# Rowan Brown, 13.12.2024
# Munich, Germany

import xarray as xr 
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 
from sea_ice_concentration import select_nearest_coord

def open_mooring_ml_data():
    """Opens CTD data in the mixed layer from the Weddell Sea mooring.
    We're interested because it seems to show a convective plume."""

    # getting part of the file path, which is saved in a text file to avoid publishing it to GitHub
    # the [0] accesses the first line, and the [:-1] removes the newline tag
    with open('../filepaths/mooring_filepath') as f: dirpath = f.readlines()[0][:-1] 

    # creating the full filepaths to the .mat files
    filepath_BGC_SBE = dirpath + '/CTD/Mooring/BGC_SBE.mat' # CTD .mat file
    filepath_sal_BGC = dirpath + '/CTD/Mooring/sal_BGC.mat' # Corrected salinities for two of the sensors

    # opening the .mat files
    mat = spio.loadmat(filepath_BGC_SBE)['SBE'] # SBE refers to Sea Bird (the instumentation company)
    mat_corr = spio.loadmat(filepath_sal_BGC)

    # extracting the needed data from the main .mat file
    jul, T, S, P  = mat['jul'][0], mat['T'][0], mat['S'][0], mat['P'][0] # un-nesting the data

    # extracting the corrected salinities and updating the S array
    Sal_449 = mat_corr['Sal_449'] #[i[0] for i in mat_corr['Sal_449']]
    Sal_2100 = mat_corr['Sal_2100'] #[i[0] for i in mat_corr['Sal_2100']]
    S[4] = Sal_449
    S[2] = Sal_2100

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

    # Adding calculation of density
    # Note: sigma0 isn't strictly correct because (the gsw package) requires conservative temperature, which we don't have 
    #       this leaves us the "pot_rho_t_exact" method/function, but then we need pressure and we only have 2 working P sensors (and I don't really trust on of them)
    #       I am therefore using the p_from_z function, which is nice except I'm also not sure how much I actually trust the depths either
    ds = ds.assign_coords(p_from_z=gsw.p_from_z(ds['depth'],-69.0005))
    ds['pot_rho'] = gsw.pot_rho_t_exact(ds['S'],ds['T'],ds['p_from_z'],0) - 1000
    
    return ds

# plotting temperature
def temp_hovm(ds):
    """Created a Hovmöller plot of temperature."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 4))
    ds.T.sel(depth=[-50,-135,-220]).plot.contourf('day','depth',ax=ax,levels=20,cbar_kwargs={'label': 'Temperature ($\degree C$)', 'orientation': 'horizontal'})
    ax.set_ylabel('Depth ($m$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.tick_params(size=9)
    ax.set_title('Temperature at the Weddell Sea mooring',fontsize=12)
    
    # Adding the sea ice data to the plot
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
    ds_si = xr.open_dataset(filepath).sel(date=slice("2021-03-26", "2022-04-06")).isel(x=id[0],y=id[1])
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:grey'
    ax2.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax2.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.savefig('Figures/Mooring_temperature_hovm_4x4.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Mooring_temperature_hovm_4x4.pdf',format='pdf',bbox_inches='tight')

# plotting salinity
def sal_hovm(ds):
    """Created a Hovmöller plot of salinity."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 4))
    p = ds.S.sel(depth=[-50, -90, -135, -170, -220]).plot.contourf('day','depth',ax=ax,levels=10,add_colorbar=False)
    ax.set_ylabel('Depth ($m$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.tick_params(size=9)
    ax.set_title('Salinity at the Weddell Sea mooring',fontsize=12)
    cbar = plt.colorbar(p, orientation="horizontal", label='Salinity ($PSU$)')
    cbar.ax.tick_params(rotation=35)

    # Adding the sea ice data to the plot
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
    ds_si = xr.open_dataset(filepath).sel(date=slice("2021-03-26", "2022-04-06")).isel(x=id[0],y=id[1])
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:grey'
    ax2.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax2.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.savefig('Figures/Mooring_salinity_hovm_4x4.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Mooring_salinity_hovm_4x4.pdf',format='pdf',bbox_inches='tight')

# plotting desnity
def rho_hovm(ds):
    """Created a Hovmöller plot of density."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 4))
    p = ds.pot_rho.sel(depth=[-50,-135,-220]).plot.contourf('day','depth',ax=ax,levels=10,add_colorbar=False,cmap=plt.colormaps['hot_r'])
    ax.set_ylabel('Depth ($m$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.tick_params(size=9)
    ax.set_title('Density at the Weddell Sea mooring',fontsize=12)
    cbar = plt.colorbar(p, orientation="horizontal", label='Potential density ($kg$ $m^{-3}$)')
    cbar.ax.tick_params(rotation=35)

    # Adding the sea ice data to the plot
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
    ds_si = xr.open_dataset(filepath).sel(date=slice("2021-03-26", "2022-04-06")).isel(x=id[0],y=id[1])
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:grey'
    ax2.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax2.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.savefig('Figures/Mooring_density_hovm_4x4.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Mooring_density_hovm_4x4.pdf',format='pdf',bbox_inches='tight')

if __name__=="__main__":
    ds = open_mooring_ml_data()
    temp_hovm(ds)
    sal_hovm(ds)
    rho_hovm(ds)