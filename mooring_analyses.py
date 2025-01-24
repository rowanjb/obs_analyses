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

def convective_resistance(ds):
    """Calculates convective resistance, ie how much heat needs to be removed to cause homogenization. 
    Reference depth is taken as 220 m, since this is the bottom working salinity sensor. 
    Note: Convective resistence /assumes/ that none of the heat loss goes into creating sea ice; maybe see Wilson and M. for more on this. 
        --->How to deal with pack ice? Some HF will make or melt ice, other will go into the water...
    """
    pot_rho, ref_depth = ds['pot_rho'], 220 # Variables needed for calculating the convective resistance 
    pot_rho = pot_rho.where(pot_rho>0,drop=True) # Dropping whereever we had a temp but no salinity and tf no rho
    pot_rho = pot_rho.assign_coords(dz=('depth',[(-1)*(pot_rho['depth'][n].values-d) for n,d in enumerate(np.append([0],pot_rho['depth'].values)[:-1])])) # Adding the 0 to get 50 at the start of the list  (ds['depth'][n]-ds['depth'][n-1])
    term2 = pot_rho * pot_rho['dz']
    convr = 9.81*(ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth'))
    return convr

def density_flux(ds):
    """/Assumes through the surface/. Also neglects absolute surface (where dS is because) because lack of data.
    Desnity flux represents the difference in mass in the water column between the stratified case and homogenized case."""

    # We'll need dz later
    # Adding the 0 to get 50 at the start of the list  (ds['depth'][n]-ds['depth'][n-1])
    dz = lambda ds : [(-1)*(ds['depth'][n].values-d) for n,d in enumerate(np.append([0],ds['depth'].values)[:-1])] 
    
    # Density/mass anomaly
    pot_rho, ref_depth = ds['pot_rho'], 220 # Variables needed for calculating the convective resistance 
    pot_rho = pot_rho.where(pot_rho>0,drop=True) # Dropping whereever we had a temp but no salinity and tf no rho
    pot_rho = pot_rho.assign_coords(dz=('depth', dz(pot_rho)))
    term2 = pot_rho * pot_rho['dz']
    dens_flux = ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth') 

    # Heat content (anomaly?)
    refT, rho_0, C_p = -1.8, 1026, 3992 # Alternative: ds['Cp'] = gsw.cp_t_exact(ds['S'],ds['T'],ds['p_from_z'])  
    T = ds['T']
    T = T.assign_coords(dz=('depth', dz(T)))
    HC = rho_0 * C_p * 10**(-9) * ((T.sel(depth=slice(0,(-1)*ref_depth))-refT)*T['dz']).sum(dim='depth') # the 10^-9 makes the result GJ

    # Salt content
    S = ds['S'].where(ds['S']>0,drop=True)
    S = S.assign_coords(dz=('depth',dz(S)))
    SC = (S*S['dz']).sum(dim='depth')
    
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(7, 4))
    color = 'tab:blue'
    dens_flux.plot(ax=ax,color=color)
    ax.set_ylabel('Mass anomaly ($kg$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.tick_params(size=9)
    ax.set_ylim(0,20)
    ax.set_title('Density changes at the Weddell Sea mooring',fontsize=12)
    ax.tick_params(size=9,color=color)
    ax.yaxis.label.set_color(color=color)
    ax.tick_params(axis='y', colors=color)

    ax2 = ax.twinx()
    color = 'tab:red'
    HC.plot(ax=ax2,color=color)
    ax2.set_ylabel('Heat content ($GJ$)',fontsize=11,color=color)
    ax2.set_xlabel('',fontsize=11)
    ax2.tick_params(size=9,color=color)
    ax2.yaxis.label.set_color(color=color)
    ax2.tick_params(axis='y', colors=color)

    ax3 = ax.twinx()
    color = 'tab:green'
    SC.plot(ax=ax3,color=color)
    ax3.set_ylabel('Salt content ($PSU$)',fontsize=11,color=color)
    ax3.set_xlabel('',fontsize=11)
    ax3.tick_params(size=9,color=color)
    ax3.yaxis.label.set_color(color=color)
    ax3.tick_params(axis='y', colors=color)
    ax3.spines.right.set_position(("axes", 1.2))

    # Adding the sea ice data to the plot
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
    ds_si = xr.open_dataset(filepath).sel(date=slice("2021-03-26", "2022-04-06")).isel(x=id[0],y=id[1])
    ax4 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:grey'
    ax4.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax4.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
    ax4.tick_params(axis='y', labelcolor=color)
    ax4.spines.right.set_position(("axes", 1.4))

    plt.savefig('density_flux.png',bbox_inches='tight',dpi=250)

    return dens_flux

# EGU:
# do the same thing for salinity and then say, based on the fluxes of salinity and heat that we have, regardless of the concentration 
# of sea ice, we know what induced overturning. Then we can use ERA5/IFS plus satellite concentrations to guess where else this
# could happen, and we can model the salt+heat flux using MITgcm (once we also have a good resolution, mixing scheme, and boundary
# conditions)

# plotting temperature
def temp_hovm(ds):
    """Created a Hovmöller plot of temperature."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 4))
    ds.T.sel(depth=[-50, -90, -135, -170, -220]).plot.contourf('day','depth',ax=ax,levels=20,cbar_kwargs={'label': 'Temperature ($\degree C$)', 'orientation': 'horizontal'})
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

    # Adding convective resistance to the plot
    convr = convective_resistance(ds) # kind of recursive...
    ax3 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax3.spines.right.set_position(("axes", 1.25))
    ax3.set_ylabel('Convective resistance ($J$ $m^{-3}$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax3.plot(convr['day'], convr, color=color, linewidth=1)
    ax3.tick_params(axis='y', labelcolor=color)

    plt.savefig('Figures/Mooring_temperature_hovm_4x4.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Mooring_temperature_hovm_4x4.pdf',format='pdf',bbox_inches='tight')

# plotting salinity
def sal_hovm(ds):
    """Created a Hovmöller plot of salinity."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 4))
    p = ds.S.sel(depth=[-50, -135, -220]).plot.contourf('day','depth',ax=ax,levels=10,add_colorbar=False)
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

    # Adding convective resistance to the plot
    convr = convective_resistance(ds) # kind of recursive...
    ax3 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax3.spines.right.set_position(("axes", 1.25))
    ax3.set_ylabel('Convective resistance ($J$ $m^{-3}$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax3.plot(convr['day'], convr, color=color, linewidth=1)
    ax3.tick_params(axis='y', labelcolor=color)

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

    # Adding convective resistance to the plot
    convr = convective_resistance(ds) # kind of recursive...
    ax3 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax3.spines.right.set_position(("axes", 1.25))
    ax3.set_ylabel('Convective resistance ($J$ $m^{-3}$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax3.plot(convr['day'], convr, color=color, linewidth=1)
    ax3.tick_params(axis='y', labelcolor=color)

    plt.savefig('Figures/Mooring_density_hovm_4x4.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Mooring_density_hovm_4x4.pdf',format='pdf',bbox_inches='tight')

if __name__=="__main__":
    ds = open_mooring_ml_data()
    density_flux(ds)
    #temp_hovm(ds)
    #sal_hovm(ds)
    #rho_hovm(ds)