# Rowan Brown, 08.2025
# Munich, Germany
# These were previously part of mooring_analyses.py but I'm separating analysis from plotting 

import xarray as xr 
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
import matplotlib.patches as ptcs
from matplotlib.colors import TwoSlopeNorm
import scipy.io as spio
import matplotlib.pyplot as plt 
from sea_ice_concentration import select_nearest_coord
import mooring_analyses as ma

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
    ax3.set_ylabel('Salt content ($g$ $kg^{-1}$)',fontsize=11,color=color)
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
    depths = {'T': [-50, -90, -125, -170, -220], 'SA': [-50, -125, -220], 'pot_rho': [-50, -125, -220]}
    titles = {'T': 'Temperature ($\degree C$)', 'SA': 'Salinity ($g$ $kg^{-1}$)', 'pot_rho': 'Potential density ($kg$ $m^{-3}$)'}
    lims = {'T': (-2,2), 'SA': (34.07, 34.91), 'pot_rho': (27.30, 27.87)}
    cm = {'T': 'coolwarm', 'SA': 'viridis', 'pot_rho': 'hot_r'}

    # Plotting
    lower_lim, upper_lim = lims[var]
    norm = plt.Normalize(lower_lim, upper_lim) # Mapping to the colourbar internal [0, 1]
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(5, 2.5))
    p = ds[var].sel(depth=depths[var]).plot.contourf('time','depth',ax=ax,levels=50,norm=norm,add_colorbar=False,cmap=plt.colormaps[cm[var]])
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
    ax3.plot(convr['time'], convr, color=color, linewidth=1)
    ax3.tick_params(axis='y', labelcolor=color, labelsize=9)
    '''

    fp = ('Figures/hovmollers/Mooring_'+var+'_hovm_'+str(start_date.year)+str(start_date.month).zfill(2)+
          str(start_date.day).zfill(2)+'-'+str(end_date.year)+str(end_date.month).zfill(2)+str(end_date.day).zfill(2)+'.png')
    plt.savefig(fp,bbox_inches='tight',dpi=900)
    print(fp)
    #plt.savefig('Figures/Mooring_temperature_hovm_4x4_short2.pdf',format='pdf',bbox_inches='tight')

# plotting temperature
def plt_hovm_EGU(ds, start_date, end_date, **kwargs):
    """*Plots used in my EGU25 poster, created specifically for posterity.
    Created a Hovmöller plot of (e.g.,) temperature.
    var is a string: "T" "SA" "pot_rho".
    Dates should be datetime objects.
    **kwargs can contain:
        an optional 'vlines' list of datetime objects
        an optional 'vlines_colour; (e.g., 'k')
        lists of parameters for a patch, i.e., [((start_x_coord, start_y_coord), thickness, height)]
            E.g., [((datetime(2021,9,13,21),-220), timedelta(hours=6), 170)]"""

    # Some var-specific definitions
    depths = {'T': [-50, -90, -125, -170, -220], 'SA': [-50, -125, -220], 'pot_rho': [-50, -125, -220]}
    titles = {'T': 'In-situ\ntemperature', 'SA': 'Absolute\nsalinity', 'pot_rho': 'Potential\ndensity'}
    units = {'T': '$\degree C$', 'SA': '$g$ $kg^{-1}$', 'pot_rho': '$kg$ $m^{-3}$'}
    lims = {'T': (-2,1), 'SA': (34.07, 34.91), 'pot_rho': (27.30, 27.87)}
    lims_short = {'T': (-1.831,0.87), 'SA': (34.6, 34.878), 'pot_rho': (27.719, 27.832)}
    cm = {'T': 'coolwarm', 'SA': 'viridis', 'pot_rho': 'hot_r'}

    #== Plotting ==#
    
    plt.rcParams["font.family"] = "serif" # change the base font
    f, axs = plt.subplots(nrows=3,ncols=1,figsize=(6, 5),sharex=True)
    
    vars = ['T','SA','pot_rho']
    for n in [0,2,1]: # We want this order for reasons
        var = vars[n]
        lower_lim, upper_lim = lims[var]
        if vars[n]=='T': norm = TwoSlopeNorm(0,lower_lim,upper_lim)
        else: norm = plt.Normalize(lower_lim, upper_lim) # Mapping to the colourbar internal [0, 1]
        p = ds[var].sel(depth=depths[var]).plot.contourf('time','depth',ax=axs[n],levels=50,norm=norm,add_colorbar=False,cmap=plt.colormaps[cm[var]])

        # Adding the sea ice data to the plot
        with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
        filepath = dirpath + '/sea_ice_concentration.nc'
        id = select_nearest_coord(longitude = -27.0048333, latitude = -69.0005000) # Note 332.9125, -69.00584 is only 3360.27 m from the mooring
        ds_si = xr.open_dataset(filepath).sel(date=slice(np.datetime64(start_date), np.datetime64(end_date))).isel(x=id[0],y=id[1])
        ax2 = axs[n].twinx()  # instantiate a second Axes that shares the same x-axis
        color = 'tab:grey'
        ax2.spines.right.set_position(("axes", 1.3))
        ax2.set_ylabel('')#Sea ice conc. ($\%$)', color=color, fontsize=11)  # we already handled the x-label with ax1
        ax2.plot(ds_si['date'], ds_si['ice_conc'][:,0,0], color=color, linewidth=1)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=9)
        
        cbar = plt.colorbar(p, orientation="vertical",format=ticker.FormatStrFormatter('%.2f'))#, label='Temperature ($\degree C$)')
        cbar.set_label(units[var], rotation=90, fontsize=9)
        cbar.ax.tick_params(labelsize=9)
        cbar.ax.set_ylim(lower_lim, upper_lim)
        axs[n].set_ylabel('')#Depth ($m$)',fontsize=11)
        axs[n].set_yticks(depths[var])
        axs[n].set_xlabel('',fontsize=11)
        axs[n].set_xlim(start_date,end_date)
        axs[n].tick_params(labelsize=9)
        axs[n].yaxis.set_major_formatter(ticker.FormatStrFormatter('%d m'))

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
        axs[n].xaxis.set_major_formatter(formatter=formatter)
        axs[n].xaxis.set_major_locator(locator=locator)

        axs[n].grid(True,c='white',lw=0.5,alpha=0.5) #xaxis.

        # vlines
        if 'vlines' in kwargs:
            if 'vlines_colour' in kwargs: c = kwargs['vlines_colour']
            else: c = 'black' #(55/256, 167/256, 222/256) # AWI colour
            for vline_date in kwargs['vlines']: 
                axs[n].vlines(vline_date,-220,-50,colors=c) 
        
        # patches
        if 'patches' in kwargs:
            for patch in kwargs['patches']:
                start_coords, width, height = patch
                rect = ptcs.Rectangle(start_coords, width, height, fc="grey", ec='grey', alpha=0.3)
                axs[n].add_patch(rect)

        # Annotation
        t = axs[n].text(0.035,0.095,titles[var],transform=axs[n].transAxes,fontsize=10)
        t.set_bbox(dict(facecolor='white', alpha=0.5,lw =0, boxstyle='square,pad=0.1'))


    axs[1].set_ylabel('Depth ($m$)',fontsize=10)
    ax2.set_ylabel('Sea ice concentration ($\%$)', color=color, fontsize=10) 

    fp = ('Figures/hovmollers/EGU_mooring_hovm_'+str(start_date.year)+str(start_date.month).zfill(2)+
          str(start_date.day).zfill(2)+'-'+str(end_date.year)+str(end_date.month).zfill(2)+str(end_date.day).zfill(2)+'.png')
    plt.savefig(fp,bbox_inches='tight',dpi=900)
    print(fp)
    #plt.savefig('Figures/Mooring_temperature_hovm_4x4_short2.pdf',format='pdf',bbox_inches='tight')

def mooring_TS(ds, start_date, end_date):
    """Plotting a TS diagram of the mooring data.
    Ultimately hoping to see if there is evidence of a front or not, and 
    also if there are interesting non-linear effects."""

    # Basic figure stuff
    plt.rcParams["font.family"] = "serif" # change the base font
    layout = [['ax1','ax2'],
              ['ax3','.'  ]]
    fig, axd = plt.subplot_mosaic(layout,figsize=(7, 7))
    ax1, ax2, ax3 = axd['ax1'], axd['ax2'], axd['ax3']
    
    ds = ds.sel(time=slice(np.datetime64(start_date), np.datetime64(end_date)))

    def plot_TS(ax,ds,d):
        SA_min, SA_max = ds['SA'].sel(depth=d).min().values-0.05, ds['SA'].sel(depth=d).max().values+0.05
        T_min, T_max = ds['T'].sel(depth=d).min().values-0.05, ds['T'].sel(depth=d).max().values+0.05
        SA_1D = np.linspace(SA_min,SA_max,50)
        T_1D = np.linspace(T_min,T_max,50)
        rho_2D = np.zeros((50,50))
        p = ds['p_from_z'].sel(depth=d).mean().values # For now we're just going to look at one depth and hence one pressure
        for col,s in enumerate(SA_1D):
            for row,t in enumerate(T_1D):
                rho_2D[row,col] = gsw.rho_t_exact(s,t,p) - 1000
        print(rho_2D)

        CS = ax.contour(SA_1D,T_1D,rho_2D,colors='k')
        ax.clabel(CS, fontsize=9)

        colours = lambda cm,ds : plt.get_cmap(cm)(np.linspace(0, 1, len(ds['time'])))
        sc = ax.scatter(ds['SA'].sel(depth=d),ds['T'].sel(depth=d),c=colours('plasma',ds),s=0.1)

        ax.set_ylabel("In situ temperature ($℃$)", fontsize=9)
        ax.set_xlabel("Absolute salinity ($g$ $kg^{-1}$)", fontsize=9)
        ax.tick_params(axis='both', labelsize=9)

        return sc, colours

    sc,colours=plot_TS(ax1,ds,-50)
    sc,colours=plot_TS(ax2,ds,-125)
    sc,colours=plot_TS(ax3,ds,-220)
    
    ax1.set_title("50 m depth", fontsize=11)
    ax2.set_title("125 m depth", fontsize=11)
    ax3.set_title("220 m depth", fontsize=11)
    plt.suptitle("Mooring sensor TS diagrams")

    # The following is almost directly from Copilot, and it handles the colourbar
    dates = [pd.Timestamp(date).to_pydatetime() for date in ds['time'].values]
    norm = plt.Normalize(dates[0].toordinal(), dates[-1].toordinal()) 
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=norm) 
    cbar_ax = fig.add_axes([0.6, 0.1, 0.025, 0.35])
    cbar = plt.colorbar(sm, ax=ax3,cax=cbar_ax)
    tick_locs = np.linspace(dates[0].toordinal(), dates[-1].toordinal(), 10)
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels([pd.Timestamp.fromordinal(int(tick)).strftime('%Y-%m-%d') for tick in tick_locs])
    #cbar.set_label('Date', fontsize=9)

    plt.tight_layout()
    plt.savefig('TS.png',dpi=1200)

def contents(ds, start_date, end_date, window=24, dt=False):
    """Similar idea to the TS diagrams, but with temp and salt time series
    Ultimately hoping to see if there is evidence of a front or not, and 
    also if there are interesting non-linear effects."""

    # Basic figure stuff
    plt.rcParams["font.family"] = "serif" # change the base fonthttps://vscode-remote+ssh-002dremote-002balbedo0.vscode-resource.vscode-cdn.net/albedo/home/robrow001/obs_analyses/contents_dt_window48.png?version%3D1754317957463

    layout = [['ax1','ax1','ax1','ax1','ax1','.'],
              ['ax1','ax1','ax1','ax1','ax1','.'],
              ['ax1','ax1','ax1','ax1','ax1','.'],
              ['ax2','ax2','ax2','ax2','ax2','.'],
              ['ax2','ax2','ax2','ax2','ax2','.'],
              ['ax2','ax2','ax2','ax2','ax2','.'],
              ['.'  ,'.'  ,'.'  ,'.'  ,'.'  ,'.']]
    fig, axd = plt.subplot_mosaic(layout,figsize=(7, 7),sharex=True)
    ax1, ax2 = axd['ax1'], axd['ax2']

    ds = ds.sel(time=slice(np.datetime64(start_date), np.datetime64(end_date)))
    
    def plotter(var,d,c,ax):
        da = ds[var].sel(depth=d).rolling(time=window, center=True).mean() # Smooth the data once so that it's not so noisy
        if dt==True: # If we want the time derivative
            da = da.diff(dim='time')/(7200/86400) # Divide by the delta t (2 hours, or 7200 seconds) and multiply by seconds-per-day (86400) to get the per-day-rate-of-change
            da = da.rolling(time=window, center=True).mean() # Take the rolling mean again to smooth the data ( can you do this? )
        p, = da.plot(ax=ax,c=c,label=str(d)+' m')
        if dt==True: # I'm realising that in this case we really want the 0 point to be equal
            ylim = abs(da).max(skipna=True)
            ax.set_ylim(-(ylim+0.05*ylim),(ylim+0.05*ylim))
            ax.hlines(0, da.time.isel(time=0), da.time.isel(time=-1), colors='k')
            ax.vlines(datetime(2021,9,7,12,0,0),-(ylim+0.05*ylim),(ylim+0.05*ylim),colors='k')
            ax.vlines(datetime(2021,9,12,8,0,0),-(ylim+0.05*ylim),(ylim+0.05*ylim),colors='k')
        if dt==False: # Only really need these means if we're no looking at derivatives
            start_slice_aug, end_slice_aug = datetime(2021,8,1,0,0,0), datetime(2021,9,1,0,0,0) # A bit ugly and hardcoded...
            start_slice_dec, end_slice_dec = datetime(2021,12,5,0,0,0), datetime(2022,1,5,0,0,0)
            ax.set_xlim(start_slice_aug-timedelta(days=20),end_slice_dec+timedelta(days=20))
            start_mean = da.sel(time=slice(np.datetime64(start_slice_aug), np.datetime64(end_slice_aug))).mean().values
            end_mean = da.sel(time=slice(np.datetime64(start_slice_dec), np.datetime64(end_slice_dec))).mean().values
            ax.text(start_slice_aug,start_mean,str(start_mean)[0:5],color=c,fontsize=9,horizontalalignment='right',verticalalignment='center')
            ax.text(end_slice_dec,end_mean,str(end_mean)[0:5],color=c,fontsize=9,horizontalalignment='left',verticalalignment='center')
            ax.hlines(start_mean, start_slice_aug, end_slice_aug, colors=c)
            ax.hlines(end_mean, start_slice_dec, end_slice_dec, colors=c)
            ymax, ymin = da.max(skipna=True), da.min(skipna=True)
            ax.set_ylim(ymin, ymax)
            ax.vlines(datetime(2021,9,7,12,0,0),ymin-0.05*ymin,ymax+0.05*ymax,colors='k')
            ax.vlines(datetime(2021,9,12,8,0,0),ymin-0.05*ymin,ymax+0.05*ymax,colors='k')
        ax.tick_params(axis='y', colors=c)
        ax.set_title("")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis='both', labelsize=9)
        ax.tick_params(axis='x', labelbottom=False)
        return p

    p1 = plotter('T',-50,'b',ax1)
    ax1a = ax1.twinx()
    p1a = plotter('T',-125,'r',ax1a)
    ax1b = ax1.twinx()
    p1b = plotter('T',-220,'g',ax1b)
    ax1b.spines['right'].set_position(("axes", 1.18))

    p2 = plotter('SA',-50,'b',ax2)
    ax2a = ax2.twinx()
    p2a = plotter('SA',-125,'r',ax2a)
    ax2b = ax2.twinx()
    p2b = plotter('SA',-220,'g',ax2b)
    ax2b.spines['right'].set_position(("axes", 1.18))

    # Calculating correlations
    def time_series_corr(ds,var,depth1,depth2,window,dt):
        da = ds[var].rolling(time=window, center=True).mean()
        if dt==True:
            da = da.diff(dim='time')/(7200/86400) # Divide by the delta t (2 hours, or 7200 seconds) and multiply by seconds-per-day (86400) to get the per-day-rate-of-change
        return xr.corr(da.sel(depth=depth1), da.sel(depth=depth2)).values 
    r_t_50_125  = time_series_corr(ds, 'T', -50,-125,window,dt) #xr.corr(ds['T'].sel(depth=-50), ds['T'].sel(depth=-125)).values
    r_t_125_220 = time_series_corr(ds, 'T',-125,-220,window,dt) #xr.corr(ds['T'].sel(depth=-125), ds['T'].sel(depth=-220)).values
    r_t_50_220  = time_series_corr(ds, 'T', -50,-220,window,dt) #xr.corr(ds['T'].sel(depth=-50), ds['T'].sel(depth=-220)).values
    r_sa_50_125 = time_series_corr(ds,'SA', -50,-125,window,dt) #xr.corr(ds['SA'].sel(depth=-50), ds['SA'].sel(depth=-125)).values
    r_sa_125_220= time_series_corr(ds,'SA',-125,-220,window,dt) #xr.corr(ds['SA'].sel(depth=-125), ds['SA'].sel(depth=-220)).values
    r_sa_50_220 = time_series_corr(ds,'SA', -50,-220,window,dt) #xr.corr(ds['SA'].sel(depth=-125), ds['SA'].sel(depth=-220)).values

    note = '*Note: Rolling means with window='+str(window*2)+' hours are used in plots and in corr. calculations'
    ann_temp = ('Pearson coeffs. (T)\n'
                '(-50 m, -125 m): '+str(r_t_50_125)[0:5]+'\n'
                '(-125 m, -220 m): '+str(r_t_125_220)[0:5]+'\n'
                '(-50 m, -220 m): '+str(r_t_50_220)[0:5])
    ann_salt = ('Pearson coeffs. (SA)\n'
                '(-50 m, -125 m): '+str(r_sa_50_125)[0:5]+'\n'
                '(-125 m, -220 m): '+str(r_sa_125_220)[0:5]+'\n'
                '(-50 m, -220 m): '+str(r_sa_50_220)[0:5])

    if dt==False:
        ax1_title = "In situ temperature ($℃$)"
        ax2_title = "Absolute salinity ($g$ $kg^{-1}$)"
        sup_title = "Mooring temperature and salintity time series"
        file_name = 'contents_window'+str(window)+'.png'
    elif dt==True:
        ax1_title = "In situ temperature rate of change ($℃$ $day^{-1}$)"
        ax2_title = "Absolute salinity rate of change ($g$ $kg^{-1}$ $day^{-1}$)"
        sup_title = "Mooring temperature and salintity rates of change time series"
        file_name = 'contents_dt_window'+str(window)+'.png'
    ax1.set_title(ax1_title, fontsize=11)
    ax2.set_title(ax2_title, fontsize=11)
    plt.suptitle(sup_title)
    ax2.tick_params(axis='x', labelbottom=True)
    fig.subplots_adjust(wspace=0, hspace=1)
    ax2.legend(handles=[p2,p2a,p2b], loc='upper center', bbox_to_anchor=(0.15,-0.15), title="Nominal sensor depth", fontsize=9, title_fontsize=9)
    ax2.text(0.8,-0.21,ann_temp,transform=ax2.transAxes,fontsize=9,verticalalignment='top')
    ax2.text(0.4,-0.21,ann_salt,transform=ax2.transAxes,fontsize=9,verticalalignment='top')
    ax2.text(-0.1,-0.7,note,transform=ax2.transAxes,fontsize=9)
    ax1.xaxis.grid(True)
    ax2.xaxis.grid(True)

    plt.savefig(file_name,dpi=1200)

def compare_CTD_cast_and_mooring(ds_mooring, ds_CTD):
    """Plot two column figure of T and S from start and end of mooring
    time series versus the launch and pickup cruise CTD casts."""

    fig, [ax1, ax2] = plt.subplots(ncols=2,nrows=1)

    # Getting the dates used in the plots
    mooring_start_id, mooring_end_id = 0,-2 #15, -15
    mooring_start_time = ds_mooring['time'].isel(time=mooring_start_id).values
    mooring_end_time = ds_mooring['time'].isel(time=mooring_end_id).values
    CTD_start_time = ds_CTD['datetime'].isel(datetime=0).values
    CTD_end_time = ds_CTD['datetime'].isel(datetime=-1).values   

    # Plotting the start mooring points
    mts = ax1.scatter(ds_mooring['T'].isel(time=15),ds_mooring['p_from_z'],
                      c='k',label='Mooring temperature start')
    mss = ax2.scatter(ds_mooring['S'].isel(time=15),ds_mooring['p_from_z'],
                      c='k',label='Mooring salinity start')

    # Plotting the end mooring points
    mte = ax1.scatter(ds_mooring['T'].isel(time=-15),ds_mooring['p_from_z'],
                      c='r',label='Mooring temperature end')
    mse = ax2.scatter(ds_mooring['S'].isel(time=-15),ds_mooring['p_from_z'],
                      c='r',label='Mooring salinity end')

    # Plotting the first CTD cast
    ctdts, = ax1.plot(ds_CTD['T'].isel(datetime=0),ds_CTD['P'],
                      c='k',label='CTD temperature start')
    ctdss, = ax2.plot(ds_CTD['S'].isel(datetime=0),ds_CTD['P'],
                      c='k',label='CTD salinity start')

    # Plotting the second CTD cast
    ctdte, = ax1.plot(ds_CTD['T'].isel(datetime=-1),ds_CTD['P'],
                      c='r',label='CTD temperature end')
    ctdse, = ax2.plot(ds_CTD['S'].isel(datetime=-1),ds_CTD['P'],
                      c='r',label="CTD salinity end")

    ax1.invert_yaxis()
    ax2.invert_yaxis()

    leg1 = ax1.legend(
        [mts,mte],
        [str(mooring_start_time)[:10],
         str(mooring_end_time)[:10]],
        title="Mooring\n(resampled daily)",
        fontsize='small',
        framealpha=0,
        title_fontsize='small',
        loc="lower left",
    )

    leg2 = ax1.legend(
        [ctdss,ctdse],
        ['PS124 ('+str(CTD_start_time)[:10]+')',
         'PS129: ('+str(CTD_end_time)[:10]+')'],
        title="CTD casts\n(launch and pick up)",
        fontsize='small',
        framealpha=0,
        title_fontsize='small',
        loc="center left",
        bbox_to_anchor = [0, 0.3]
    )

    leg1._legend_box.align = "left"
    leg2._legend_box.align = "left"

    ax1.add_artist(leg1)
    ax1.add_artist(leg2)

    plt.suptitle("Mooring instruments vs CTD casts")
    ax1.set_title("Temperature")
    ax2.set_title("Salinity")
    ax1.set_xlabel("In situ temperature ($℃$)")
    ax1.set_ylabel("Pressure ($dbar$)")
    ax2.set_xlabel("Practical salinity ($PSU$)")

    plt.savefig('mooring_vs_CTD.png',dpi=600)

if __name__=="__main__":   
    ds = ma.open_mooring_data().convert_to_daily()
    ds.append_gsw_vars()
    ds_CTD = ma.open_mooring_profiles_data()
    #compare_CTD_cast_and_mooring(ds, ds_CTD)
    #ds = ma.correct_mooring_salinities(ds)
    #start_date, end_date = datetime(2021,4,1,0,0,0), datetime(2022,4,1,0,0,0)
    #mooring_TS(ds, start_date, end_date)
    #start_date, end_date = datetime(2021,9,6,0,0,0), datetime(2021,9,17,0,0,0)
    #for w in [  1,  3,  6, 12, 24, 48, 96,192,384]:
    #    contents(ds, start_date, end_date, window=w, dt=True)
    #start_date, end_date = datetime(2021,8,1,0,0,0), datetime(2022,1,5,0,0,0)
    #for w in [  1,  3,  6, 12, 24, 48, 96,192,384]:
    #    contents(ds, start_date, end_date, window=w, dt=False)