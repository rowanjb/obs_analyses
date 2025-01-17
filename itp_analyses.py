# Rowan Brown, 12.12.2024
# Munich, Germany

import os
import xarray as xr 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
import matplotlib.path as mpath
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import matplotlib.colors as colors
import numpy as np

def open_dat_file(itp):
    """Opens level 3 pressure-bin-averaged data at 1-db vertical resolution.
    All ITPs should have associated .dat files in the grddata directory, which holds the level-3 "corrected" data.
    Note I found ITP132 doesn't have salinities in the cleaned .mat file, which is why I'm using the .dat files."""

    with open('../filepaths/itp_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    all_files = sorted(os.listdir(dirpath+str(itp)+'/itp'+str(itp)+'grddata/')) # list of files in the dir
    ps, ts, ss, dates, lons, lats = np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]) # init lists
    for n,file in enumerate(all_files): # loop through the files
        filepath = dirpath+str(itp)+'/itp'+str(itp)+'grddata/'+file 
        try: # try-except helps avoid breaking the code when opening a "bad" file
            datapd = pd.read_csv(filepath, sep='\s+', skiprows=2, skipfooter=1, engine='python') # open the fleisch of the data
            datapd = datapd.sort_values(by='day', ascending=True, ignore_index=True) # sort it so that it's time-increasing 
            locpd = pd.read_csv(filepath, sep='\s+', nrows=2, engine='python') # top of CSV has the location
            # format the date (subtract 1 from the day because Jan 1 is indexed from 1, so Jan 1 should correspond to timedelta(0 days))
            date = [datetime(datapd['%year'][n],1,1,0,0,0,0) + timedelta(datapd['day'][n]-1) for n,i in enumerate(datapd['%year'])] 
            p, t, s = datapd['pressure(dbar)'].values, datapd['temperature(C)'].values, datapd['salinity'].values # ctd data
            lon, lat = np.full(len(t),locpd.iloc[0,2]), np.full(len(t),locpd.iloc[0,3]) # expand locations to full-length lists
            ps, ts, ss = np.append(ps,p,axis=0), np.append(ts,t,axis=0), np.append(ss,s,axis=0) # save the data
            dates, lons, lats = np.append(dates,date,axis=0), np.append(lons,lon,axis=0), np.append(lats,lat,axis=0)
        except: 
            print('Bad file: ' + filepath)
            continue

    # convert strings to floats
    lons = lons.astype(float)
    lats = lats.astype(float)

    # now the data is saved in a series of very long 1D lists
    # it isn't usable now, so let's put it into a Dataset and "regularize" it
    ds = xr.Dataset(
            data_vars=dict(
                temperature=(['date'], ts),
                salinity=(['date'], ss),
                pressure=(['date'], ps),
                lon=(['date'], lons),
                lat=(['date'], lats),
            ),
            coords=dict(
                date=dates,
            ),
            attrs=dict(description='Test desc.'), # fill this out properly if saving the .nc
        )

    # binning the pressures--likely unecessary but avoids any issues relating to abnormal pressure values
    #pressure_levels = np.arange(1, 2001, 2) # down to 2000 meters (no way any ITP is deeper than this!)
    #pressure_bins = [i for i in pd.cut(ds['pressure'], bins=pressure_levels, labels=pressure_levels[:-1]+1)]
    #ds['pressure_bins'] = xr.DataArray(pressure_bins, dims='date') # add it as a variable
    #ds = ds.drop_vars('pressure') # remove the old pressure var; it's now uneeded 

    # resample the variables in time according to grouped pressures (thanks copilot!)
    # check that this is correct if you're going to use this seriously
    ds_grouped_resampled = ds.groupby('pressure').apply(lambda bin_group : bin_group.resample(date='1d',skipna=True).mean(skipna=True))

    ds_grouped_resampled = ds_grouped_resampled.set_coords(['pressure']) # convert vars into coordinates
    ds_2d = ds_grouped_resampled.set_index(z=('date','pressure')).drop_duplicates('z').unstack('z') 

    return ds_2d

def temp_hovm(ds,itp,dlim):
    """Created a Hovmöller plot of temperature and pressure."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 3)) 
    cmap = plt.colormaps['coolwarm']
    cax = ds['temperature'].plot.contourf('date','pressure',ax=ax,levels=20,add_colorbar=False,cmap=cmap)
    cbar = f.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel('Temperature ($\degree C$)')
    ax.set_ylim(0,dlim)
    ax.grid(True)
    ax.set_ylabel('Pressure ($dbar$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.invert_yaxis()
    ax.tick_params(axis='both',labelsize=9)
    ax.set_title('Temperature for ITP '+str(itp),fontsize=12)
    f.tight_layout()    
    plt.savefig('Figures/ITP_'+str(itp)+'_temperature_hovm_4x3.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/ITP_'+str(itp)+'_temperature_hovm_4x3.pdf',format='pdf',bbox_inches='tight')


def salt_hovm(ds,itp,dlim):
    """Created a Hovmöller plot of salinity and pressure."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f, ax = plt.subplots(figsize=(4, 3)) 
    cmap = plt.colormaps['viridis']
    cax = ds['salinity'].plot.contourf('date','pressure',ax=ax,levels=20,add_colorbar=False,cmap=cmap)
    cbar = f.colorbar(cax, ax=ax)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel('Salinity ($PSU$)')
    ax.set_ylim(0,dlim)
    ax.grid(True)
    ax.set_ylabel('Pressure ($dbar$)',fontsize=11)
    ax.set_xlabel('',fontsize=11)
    ax.invert_yaxis()
    ax.tick_params(axis='both',labelsize=9)
    ax.set_title('Salinity for ITP '+str(itp),fontsize=12)
    f.tight_layout()    
    plt.savefig('Figures/ITP_'+str(itp)+'_salinity_hovm_4x3.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/ITP_'+str(itp)+'_salinity_hovm_4x3.pdf',format='pdf',bbox_inches='tight')

def temp_hovm_with_map(ds,itp,dlim):
    """Created a Hovmöller plot of temperature and pressure with a map."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f = plt.figure(figsize=(4,4.5)) #2,1,figsize=(4, 6),subplot_kw=dict(projection=projection)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    
    southLat, northLat, westLon, eastLon = -85, -55, 179, -179
    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='black')
    projection = ccrs.PlateCarree(central_longitude=(westLon+eastLon)/2)#ccrs.AlbersEqualArea(central_longitude=(westLon+eastLon)/2, central_latitude=(southLat+northLat)/2,standard_parallels=(southLat,northLat))
    subplot_kw=dict(projection=projection,aspect=4)
    ax1 = f.add_subplot(gs[0], **subplot_kw)
    ax1.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax1.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax1.coastlines(resolution='50m')
    ax1.scatter(ds['lon'],ds['lat'],marker='.',s=5,facecolors='red', edgecolors='red')
    gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5)
    gl.top_labels=False #suppress top labels
    gl.right_labels=False #suppress right labels
    gl.rotate_labels=False
    gl.ylocator = mticker.FixedLocator([-85, -80, -75, -70, -65, -60, -55])  #[50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax1.set_title("Location",fontsize=11)

    ax2 = f.add_subplot(gs[1])
    cmap = plt.colormaps['coolwarm']
    cax = ds['temperature'].plot.contourf('date','pressure',ax=ax2,levels=20,add_colorbar=False,cmap=cmap)
    cbar = f.colorbar(cax, ax=ax2)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel('Temperature ($\degree C$)')
    ax2.set_ylim(0,dlim)
    ax2.grid(True)
    ax2.set_ylabel('Pressure ($dbar$)',fontsize=11)
    ax2.set_xlabel('',fontsize=11)
    ax2.invert_yaxis()
    ax2.tick_params(axis='both',labelsize=9)
    ax2.set_title("Temperature",fontsize=11)

    f.suptitle('ITP '+str(itp),fontsize=12)
    f.tight_layout()    
    plt.savefig('Figures/ITP_'+str(itp)+'_temperature_hovm_4x3.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/ITP_'+str(itp)+'_temperature_hovm_4x3.pdf',format='pdf',bbox_inches='tight')

def salt_hovm_with_map(ds,itp,dlim):
    """Created a Hovmöller plot of salinity and pressure with a map."""
    plt.rcParams["font.family"] = "serif" # change the base font
    f = plt.figure(figsize=(4,4.5)) #2,1,figsize=(4, 6),subplot_kw=dict(projection=projection)) 
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 2])
    
    southLat, northLat, westLon, eastLon = -85, -55, 179, -179
    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='black')
    projection = ccrs.PlateCarree(central_longitude=(westLon+eastLon)/2)#ccrs.AlbersEqualArea(central_longitude=(westLon+eastLon)/2, central_latitude=(southLat+northLat)/2,standard_parallels=(southLat,northLat))
    subplot_kw=dict(projection=projection,aspect=4)
    ax1 = f.add_subplot(gs[0], **subplot_kw)
    ax1.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax1.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax1.coastlines(resolution='50m')
    ax1.scatter(ds['lon'],ds['lat'],marker='.',s=5,facecolors='red', edgecolors='red')
    gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5)
    gl.top_labels=False #suppress top labels
    gl.right_labels=False #suppress right labels
    gl.rotate_labels=False
    gl.ylocator = mticker.FixedLocator([-85, -80, -75, -70, -65, -60, -55])  #[50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax1.set_title("Location",fontsize=11)

    ax2 = f.add_subplot(gs[1])
    cmap = plt.colormaps['viridis']
    cax = ds['salinity'].plot.contourf('date','pressure',ax=ax2,levels=20,add_colorbar=False,cmap=cmap)
    cbar = f.colorbar(cax, ax=ax2)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel('Salinity ($PSU$)')
    ax2.set_ylim(0,dlim)
    ax2.grid(True)
    ax2.set_ylabel('Pressure ($dbar$)',fontsize=11)
    ax2.set_xlabel('',fontsize=11)
    ax2.invert_yaxis()
    ax2.tick_params(axis='both',labelsize=9)
    ax2.set_title("Salinity",fontsize=11)

    f.suptitle('ITP '+str(itp),fontsize=12)
    f.tight_layout()    
    plt.savefig('Figures/ITP_'+str(itp)+'_salinity_hovm_4x3.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/ITP_'+str(itp)+'_salinity_hovm_4x3.pdf',format='pdf',bbox_inches='tight')

if __name__=="__main__":
    itp = 31
    dlim = 700
    ds = open_dat_file(itp)
    #temp_hovm(ds,itp,dlim)    
    #salt_hovm(ds,itp,dlim)  
    temp_hovm_with_map(ds,itp,dlim)
    salt_hovm_with_map(ds,itp,dlim)

    itp = 40
    dlim = 425
    ds = open_dat_file(itp)
    #temp_hovm(ds,itp,dlim)
    #salt_hovm(ds,itp,dlim)
    temp_hovm_with_map(ds,itp,dlim)
    salt_hovm_with_map(ds,itp,dlim)

    itp = 132
    dlim = 250
    ds = open_dat_file(itp)
    #temp_hovm(ds,itp,dlim)
    #salt_hovm(ds,itp,dlim)
    temp_hovm_with_map(ds,itp,dlim)
    salt_hovm_with_map(ds,itp,dlim)