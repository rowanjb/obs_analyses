# Rowan Brown, 13.12.2024
# Munich, Germany

import xarray as xr 
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.patches as ptcs
import scipy.io as spio
import matplotlib.pyplot as plt 
from sea_ice_concentration import select_nearest_coord

import sys
sys.path.insert(1, '../model_analyses/')
import cell_thickness_calculator as ctc

def open_mooring_ml_data(time_delta='day'):
    """Opens CTD data in the mixed layer from the Weddell Sea mooring.
    We're interested because it seems to show a convective plume.
    The salinities returned are the corrected salinities from Carsten and Markus.
    These seem unphysical to me and can be further corrected using correct_mooring_salinities(ds).
    time_delta controls if the data is binned by 'day' or 'hour'."""

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
    hours = list(range(0,24))
    start_date = datetime(2020,12,31,0,0,0) # I'm basically just assuming this is the start date
    new_datetime_coords = []
    for day in new_day_coord:
        if time_delta=='hour':
            for hour in hours:
                new_datetime_coords.append( start_date + timedelta(days=day) + timedelta(hours=hour) )
        elif time_delta=='day':
            new_datetime_coords.append( start_date + timedelta(days=day) )
        else: 
            "Invalid time_delta (should be 'day' or 'hour')"
        
    # locally-defined function for getting the time-weighted average of a variable over one day
    def daily_avg_mooring_data(inmat): 
        
        # initializing a nan array, size is [number of days] x [number of depths]
        out = np.empty((len(new_datetime_coords),6)) 
        out[:] = np.nan

        # looping through the measured timesteps
        for i in range(len(jul)):
            if len(inmat[i]) > 0: # basically if we have a non-empty array
                date = pd.to_datetime([start_date + timedelta(days=i[0]) for i in jul[i]]) # creating datetimes for each entry
                df = pd.DataFrame(data={'date': date, 'var': [i[0] for i in inmat[i]]}) # filling a pandas dataframe
                df['weights'] = [i.total_seconds() for i in df.diff().date] # seconds elapsed between measurements
                df = df.set_index('date') 
                df = df[(df.index > '2021-03-26') & (df.index < '2022-04-07')] # truncating start and end dates 
                # taking the time-weighted average every 24 hours (or some other period)
                # weights don't align perfectly with each day, but it's close enough
                new_df = pd.DataFrame(index=new_datetime_coords) # Init the dataframe
                if time_delta=='day': 
                    new_df['var'] = df.resample('D').apply(lambda df : np.sum(df['weights']*df['var'])/np.sum(df['weights']))
                elif time_delta=='hour': 
                    new_df['var'] = df.resample('h').apply(lambda df : np.sum(df['weights']*df['var'])/np.sum(df['weights']))
                else: 
                    "Invalid time_delta (should be 'day' or 'hour')"
                out[:,i] = np.array(new_df['var']) # writing to the "out" array
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
    ds = ds.assign_coords(p_from_z=gsw.p_from_z(ds['depth'],-69.0005))
    ds['SA'] = gsw.SA_from_SP(ds['S'],ds['p_from_z'],lon=-27.0048,lat=-69.0005)
    ds['pot_rho'] = gsw.pot_rho_t_exact(ds['SA'],ds['T'],ds['p_from_z'],0) - 1000
    
    print("Mooring data opened")
    return ds

def correct_mooring_salinities(ds_mooring):
    """The salinities from the lower two sensors seem, basically, wrong, so here I'm equating the sensors' means with those of WOA
    climatologies to see if we can't maintain the interesting anomaly in the middle sensor but also attain a reasoanble mean."""

    # Extract one year of data from the mooring
    ds_mooring = ds_mooring.sel(day=slice('2021-04-01','2022-03-31'))

    # Calculate the mean salinites at the two "bad sensors"
    S = ds_mooring['S'] # Extract as a dataarray for easy handling
    S_srfce_mean_mooring = S.sel(depth=-50).mean(dim='day').values
    S_upper_mean_mooring = S.sel(depth=-135).mean(dim='day').values
    S_lower_mean_mooring = S.sel(depth=-220).mean(dim='day').values

    # Open the WOA data
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] 
    ds_woa = xr.open_dataset(dirpath + '/WOA_monthly_'+'s'+'_'+str(2015)+'.nc',decode_times=False)
    ds_woa = ds_woa.rename({'time':'month'}) 
    ds_woa['month'] = ds_woa['month'] - 35.5 # months are saved as arbitrary integers 

    # Calculate the yearly average, based on: https://docs.xarray.dev/en/latest/examples/area_weighted_temperature.html
    ds_woa['weights'] = ('month',[31,28,31,30,31,30,31,31,30,31,30,31]) # Number of days per month
    ds_woa_weighted = ds_woa['s_an'].weighted(ds_woa['weights'])
    woa_weighted_mean = ds_woa_weighted.mean('month')

    '''
    fig, ax = plt.subplots()
    ax.plot(woa_weighted_mean,(-1)*woa_weighted_mean['depth'])
    ax.plot(ds_woa['s_an'].mean(dim='month'),(-1)*woa_weighted_mean['depth'],c='r')
    plt.savefig('test.png')
    '''

    S_srfce_mean_woa = woa_weighted_mean.interp(depth=50).values
    S_upper_mean_woa = woa_weighted_mean.interp(depth=135).values
    S_lower_mean_woa = woa_weighted_mean.interp(depth=220).values

    print(" 50 m 12-month mean from mooring: "+str(S_srfce_mean_mooring))
    print(" 50 m 12-month mean from WOA:     "+str(S_srfce_mean_woa))
    S_srfce_mean_anomaly = S_srfce_mean_mooring-S_srfce_mean_woa
    print("Difference: "+str(S_srfce_mean_anomaly))
    print('')
    print("135 m 12-month mean from mooring: "+str(S_upper_mean_mooring))
    print("135 m 12-month mean from WOA:     "+str(S_upper_mean_woa))
    S_upper_mean_anomaly = S_upper_mean_mooring-S_upper_mean_woa
    print("Difference: "+str(S_upper_mean_anomaly))
    print('')
    print("220 m 12-month mean from mooring: "+str(S_lower_mean_mooring))
    print("220 m 12-month mean from WOA:     "+str(S_lower_mean_woa))
    S_lower_mean_anomaly = S_lower_mean_mooring-S_lower_mean_woa
    print("Difference: "+str(S_lower_mean_anomaly))

    # Correcting the salinities
    S = xr.where(S['depth']==-50,S.sel(depth=-50) - S_srfce_mean_anomaly,S)
    S = xr.where(S['depth']==-135,S.sel(depth=-135) - S_upper_mean_anomaly,S)
    S = xr.where(S['depth']==-220,S.sel(depth=-220) - S_lower_mean_anomaly,S)
    ds_mooring['S'] = S # Reassign the corrected values
    ##print(ds_mooring)

    # Adding calculation of density (redoing earilier calculation in above func)
    ds_mooring = ds_mooring.assign_coords(p_from_z=gsw.p_from_z(ds_mooring['depth'],-69.0005))
    ds_mooring['SA'] = gsw.SA_from_SP(ds_mooring['S'],ds_mooring['p_from_z'],lon=-27.0048,lat=-69.0005)
    ds_mooring['pot_rho'] = gsw.pot_rho_t_exact(ds_mooring['SA'],ds_mooring['T'],ds_mooring['p_from_z'],0) - 1000

    print("Salinities corrected")
    return ds_mooring

def fill_mooring_with_WOA(ds, season='autumn'):
    """For filling in a len(1) in time mooring dataset with WOA climatologies.
    season is relative to the N.H.; default is autumn (S.H. spring)."""

    # Select just the desired depth levels, if not already done
    ds = ds.where(~np.isnan(ds['pot_rho']),drop=True)

    # Calculating thickness levels
    # You can change some of these parameters if you want
    # But these are the values that I've used most commonly in the model
    depth = 500
    num_levels, numAx1, numAx2 = 50, 150, 150
    size = str(num_levels) +'x' + str(numAx1) + 'x' + str(numAx2)
    pot_temp = True # Whether you want pot or in-situ temp
    abs_salt = True # Whether you want abs or practical salt
    x1, x2 = 1, num_levels # Indices of top and bottom cells
    fx1 = 1 # Depth of bottom of top cell
    min_slope = 1 # Minimum slope (should probably > x1)
    A, B, C, _, _ = ctc.find_parameters(x1, x2, fx1, depth, min_slope)
    dz = ctc.return_cell_thicknesses(x1, x2, depth, A, B, C) 

    # Depths used in the model (calculated to the centre of the cells)
    z = np.zeros(len(dz))
    for i,n in enumerate(dz): # Getting sell depths
        if i==0: z[i] = n/2
        else: z[i] = np.sum(dz[:i]) + n/2

    # Opening the WOA data; seasons are ['winter', 'spring', 'summer', 'autumn'] (i.e., NORTHERN HEMISPHERE SEASONS!)
    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    das = xr.open_dataset(dirpath + '/WOA_seasonally_'+'s'+'_'+str(2015)+'.nc',decode_times=False)['s_an']
    s_woa = das.isel(time=2).interp(depth=z) # time=2 refers to "summer"
    dat = xr.open_dataset(dirpath + '/WOA_seasonally_'+'t'+'_'+str(2015)+'.nc',decode_times=False)['t_an']
    t_woa = dat.isel(time=2).interp(depth=z) # time=2 refers to "summer"
    p = gsw.p_from_z((-1)*z,lat=-69.0005) # Calculating pressure from depth, then getting absolute salinity, and potential temperature 
    SA = gsw.SA_from_SP(s_woa,p,lat=-69.0005,lon=-27.0048)           # ...(you should want theta/pt---this is what the model demands!)
    pt = gsw.pt0_from_t(SA,t_woa,p)

    # Determining which salt and temp to use
    if pot_temp: # If potential temp is what we're looking for, then...
        dst = gsw.pt0_from_t(ds['SA'],ds['T'],ds['p_from_z']).values # Let t (mooring) be potential temperature
        t_woa = pt # Let t (WOA) now be potential temperature 
        t_name = 'theta' # Let the var name in the file be theta
    else: # i.e., if we /don't/ want potential temp, we will use in-situ
        dst = ds['T'].values 
        t_name = 'T'
    if abs_salt: # Similarly, if it is absolute salinity that we're looking for, then...
        dss = ds['SA'].values
        s_woa = SA # And let s (WOA) be absolute salinity
        s_name = 'SA'
    else: # i.e., if we /don't/ want absolute salinity, then we likely want PSU
        dss = ds['S'].values
        s_name = 'S'
    
    # Finding depth threshold indices, i.e., where in the model depths do mooring data apply
    id50  = np.where(z == np.min(z[z>50]) )
    id135 = np.where(z == np.min(z[z>135]) )
    id220 = np.where(z == np.min(z[z>220]) )

    # Interpolating/filling values
    s, t = np.empty(len(z)), np.empty(len(z)) # These are our final s and t vectors
    for n,d in enumerate(z):
        if d<50:
            mean_diff_s = dss[0] - s_woa[id50]
            mean_diff_t = dst[0] - t_woa[id50]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
        elif d<135:
            del_s = dss[1] - dss[0]
            del_t = dst[1] - dst[0]
            weight = (d-50)/(135-50)
            s[n] = dss[0] + del_s*weight
            t[n] = dst[0] + del_t*weight
        elif d<220:
            del_s = dss[2] - dss[1]
            del_t = dst[2] - dst[1]
            weight = (d-135)/(220-135)
            s[n] = dss[1] + del_s*weight
            t[n] = dst[1] + del_t*weight
        else:
            mean_diff_s = dss[2] - s_woa[id220]
            mean_diff_t = dst[2] - t_woa[id220]
            s[n] = s_woa[n] + mean_diff_s
            t[n] = t_woa[n] + mean_diff_t
    
    # Creating an xr dataset
    ds_filled = xr.Dataset(
        data_vars=dict(
            T=(["z"], t),
            S=(["z"], s),
            dz=(["z"], dz),
        ),
        coords=dict(
            z=z,
        ),
        attrs=dict(description="Mooring data, filled with WOA climatologies and at the model's vertical resolution"),
    )

    return ds_filled

def open_mooring_profiles_data():
    """Opens CTD data from profiles taken during the mooring launch/pickup cruises.
    Work in progress..."""

    # getting part of the file path, which is saved in a text file to avoid publishing it to GitHub
    # the [0] accesses the first line, and the [:-1] removes the newline tag
    with open('../filepaths/mooring_filepath') as f: dirpath = f.readlines()[0][:-1] 
    
    # creating the full filepaths to the .mat files
    filepath_117_1 = dirpath + '/CTD/Profiles/117_1.mat'
    filepath_dPS129_072_01 = dirpath + '/CTD/Profiles/dPS129_072_01.mat'

    # opening the .mat files
    mat_124 = spio.loadmat(filepath_117_1)['S']
    mat_129 = spio.loadmat(filepath_dPS129_072_01)

    # extracting metadata
    cruise_124, lat_124, lon_124, date_124 = mat_124['CRUISE'][0][0][0], mat_124['LAT'][0][0][0][0], mat_124['LON'][0][0][0][0], mat_124['DATETIME'][0][0][0]
    cruise_129, lat_129, lon_129, date_129 = mat_129['HDR']['CRUISE'][0][0][0], mat_129['HDR']['LAT'][0][0][0][0], mat_129['HDR']['LON'][0][0][0][0], mat_129['HDR']['DATETIME'][0][0][0]
    
    unnest = lambda mat : [i[0] for i in mat]

    # extracting the hydro data 
    pres_124  = unnest(mat_124['PRES'][0][0])
    print(pres_124)
    quit()

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

    #THIS SHOULD USE SA NOT S
    #ds = ds.assign_coords(p_from_z=gsw.p_from_z(ds['depth'],-69.0005))
    #ds['pot_rho'] = gsw.pot_rho_t_exact(ds['S'],ds['T'],ds['p_from_z'],0) - 1000
    
    return ds

def convective_resistance(ds,type='heat'):
    """Calculates convective resistance, i.e., how much heat or mass needs to be removed to cause homogenization of the water column. 
    Reference depth is taken as 220 m, since this is the bottom working salinity sensor. 
    Note: Convective resistence /assumes/ that none of the heat loss goes into creating sea ice; maybe see Wilson and M. for more on this. 
        --->How to deal with pack ice? Some HF will make or melt ice, other will go into the water...
    """
    pot_rho, ref_depth = ds['pot_rho'], 220 # Variables needed for calculating the convective resistance 
    pot_rho = pot_rho.where(pot_rho>0,drop=True) # Dropping whereever we had a temp but no salinity and tf no rho
    pot_rho = pot_rho.assign_coords(dz=('depth',[(-1)*(pot_rho['depth'][n].values-d) for n,d in enumerate(np.append([0],pot_rho['depth'].values)[:-1])])) # Adding the 0 to get 50 at the start of the list  (ds['depth'][n]-ds['depth'][n-1])
    term2 = pot_rho * pot_rho['dz']
    if type=='heat':
        convr = 9.81*(ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth')) # Unit ends up being J/m3... I think
    elif type=='mass':
        convr = (ref_depth * pot_rho.sel(depth=(-1)*ref_depth) - term2.sum(dim='depth')) # Unit ends up being kg/m2
    else:
        print('Need to choose type="heat" or "mass"')
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

    plt.savefig('Figures/Density_flux.png',bbox_inches='tight',dpi=250)

    return dens_flux

# plotting temperature
def plt_hovm(ds, var, start_date, end_date, **kwargs):
    """Created a Hovmöller plot of (e.g.,) temperature.
    var is a string: "T" "SA" "pot_rho".
    Dates should be datetime objects.
    **kwargs can contain:
        an optional 'vlines' list of datetime objects
        an optional 'vlines_colour; (e.g., 'k')
        lists of parameters for a patch, i.e., [((start_x_coord, start_y_coord), thickness, height)]
            E.g., [((datetime(2021,9,13,21),-220), timedelta(hours=6), 170)]"""

    # Some var-specific definitions
    depths = {'T': [-50, -90, -135, -170, -220], 'SA': [-50, -135, -220], 'pot_rho': [-50, -135, -220]}
    titles = {'T': 'Temperature ($\degree C$)', 'SA': 'Salinity ($PSU$)', 'pot_rho': 'Potential density ($kg$ $m^{-3}$)'}
    lims = {'T': (-2,2), 'SA': (34.07, 34.91), 'pot_rho': (27.30, 27.87)}
    cm = {'T': 'coolwarm', 'SA': 'viridis', 'pot_rho': 'hot_r'}

    # Plotting
    lower_lim, upper_lim = lims[var]
    norm = plt.Normalize(lower_lim, upper_lim) # Mapping to the colourbar internal [0, 1]
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(5, 2.5))
    p = ds[var].sel(depth=depths[var]).plot.contourf('day','depth',ax=ax,levels=50,norm=norm,add_colorbar=False,cmap=plt.colormaps[cm[var]])
    ax.set_ylabel('Depth ($m$)',fontsize=11)
    ax.set_yticks(depths[var])
    ax.set_xlabel('',fontsize=11)
    ax.set_xlim(start_date,end_date)
    ax.tick_params(labelsize=9)
    ax.set_title(titles[var],fontsize=12)
    cbar = plt.colorbar(p, orientation="vertical")#, label='Temperature ($\degree C$)')
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylim(lower_lim, upper_lim)

    # Handling the xaxis formatting
    time_delta = start_date - end_date
    if abs(int(time_delta.days)) < 12: # i.e., less than 1.5 weeks-ish
        locator = mdates.DayLocator(interval=2) #WeekdayLocator(interval=2)
        formatter = mdates.DateFormatter('%d/%m')
    elif abs(int(time_delta.days)) < 32: # i.e., less than one month
        locator = mdates.WeekdayLocator(interval=7)
        formatter = mdates.DateFormatter('%d/%m')
    elif abs(int(time_delta.days)) < 190: # i.e., less than six months
        locator = mdates.MonthLocator()
        formatter = mdates.DateFormatter('%m/%y')
    else: # i.e., over six months (up to around 1 year, which is how much data we have)
        locator = mdates.MonthLocator(interval=2)
        formatter = mdates.DateFormatter('%m/%y')
    ax.xaxis.set_major_formatter(formatter=formatter)
    ax.xaxis.set_major_locator(locator=locator)

    # vlines
    if 'vlines' in kwargs:
        if 'vlines_colour' in kwargs: c = kwargs['vlines_colour']
        else: c = 'k'
        for vline_date in kwargs['vlines']: 
            ax.vlines(vline_date,-220,-50,colors=c) 
    
    # patches
    if 'patches' in kwargs:
        for patch in kwargs['patches']:
            start_coords, width, height = patch
            rect = ptcs.Rectangle(start_coords, width, height, fc="grey", ec='grey', alpha=0.3)
            ax.add_patch(rect)

    # Adding the sea ice data to the plot
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
    ds_si = xr.open_dataset(filepath).sel(date=slice(np.datetime64(start_date), np.datetime64(end_date))).isel(x=id[0],y=id[1])
    ax2 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:grey'
    ax2.spines.right.set_position(("axes", 1.3))
    ax2.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax2.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=9)

    ''' Commenting this out because I don't really like it for EGU
    # Adding convective resistance to the plot
    convr = convective_resistance(ds,type='mass') # kind of recursive...
    ax3 = ax.twinx()  # instantiate a second Axes that shares the same x-axis
    color = 'tab:red'
    ax3.spines.right.set_position(("axes", 1.6))
    ax3.set_ylabel('Mass anomaly ($kg$ $m^{-2}$)', color=color, fontsize=11)  # we already handled the x-label with ax1
    ax3.plot(convr['day'], convr, color=color, linewidth=1)
    ax3.tick_params(axis='y', labelcolor=color, labelsize=9)
    '''

    fp = ('Figures/hovmollers/Mooring_'+var+'_hovm_'+str(start_date.year)+str(start_date.month).zfill(2)+
          str(start_date.day).zfill(2)+'-'+str(end_date.year)+str(end_date.month).zfill(2)+str(end_date.day).zfill(2)+'.png')
    plt.savefig(fp,bbox_inches='tight',dpi=900)
    print(fp)
    #plt.savefig('Figures/Mooring_temperature_hovm_4x4_short2.pdf',format='pdf',bbox_inches='tight')

if __name__=="__main__":   
    ds = open_mooring_ml_data(time_delta='hour')
    ds = correct_mooring_salinities(ds).isel(day=slice(0,-1,2))

    start_date, end_date = datetime(2021,4,1,0,0,0), datetime(2022,3,31,0,0,0)
    vlines = [datetime(2021,9,10,0,0,0), datetime(2021,9,20,0,0,0)]
    plt_hovm(ds, 'T', start_date, end_date, vlines=vlines)
    plt_hovm(ds, 'SA', start_date, end_date, vlines=vlines)
    plt_hovm(ds, 'pot_rho', start_date, end_date, vlines=vlines)

    start_date, end_date = datetime(2021,9,10,0,0,0), datetime(2021,9,20,0,0,0)
    patches = [((datetime(2021,9,13,21),-220), timedelta(hours=6), 170), ((datetime(2021,9,15,21),-220), timedelta(hours=6), 170)]
    plt_hovm(ds, 'T', start_date, end_date, patches=patches)
    plt_hovm(ds, 'SA', start_date, end_date, patches=patches)
    plt_hovm(ds, 'pot_rho', start_date, end_date, patches=patches)