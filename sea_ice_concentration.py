# Rowan Brown, 17.01.2024
# Munich, Germany

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.gridspec as gridspec
import scipy.io as spio
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import subprocess
import os
import cartopy.crs as ccrs
import cartopy.feature as feature
import matplotlib.ticker as mticker
import time

def download_data(date_str):
    """Credit for much of this function and the Bash script that it calls goes to copilot.
    They work together to download hdf (version 4, I think) files of daily sea ice concentration in Antarctica."""

    # Construct the path to the given date's file
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filename = dirpath + '/concentration_data/' + 'asi-AMSR2-s6250-' + date_str + '-v5.4.hdf'
    
    # Check if the file alread exits
    if os.path.isfile(filename)==False:
        # If it doesn't exist, then download it using the bash script with the date parameter
        result = subprocess.run(['./download_conc_data.sh', date_str, filename], capture_output=True, text=True) 
        if result.returncode == 0:
            print(result.stdout)
            print("Script executed successfully")
        else:
            print("Script execution failed")
    # If it already exists, don't need to download it

def list_of_date_strs(start_date_str, end_date_str):
    """Creates a list of dates (strings) between to given dates."""
    start_date = datetime.strptime(start_date_str, '%Y%m%d')
    end_date = datetime.strptime(end_date_str, '%Y%m%d')
    all_dates = [start_date + timedelta(days=x) for x in range((end_date-start_date).days + 1)]
    all_dates_str = [date.strftime('%Y%m%d') for date in all_dates]
    return all_dates_str, all_dates

def sea_ice_conc_nc(start_date_str, end_date_str):
    """Creates .nc of daily sea ice concentration from AWI ice portal .hdf files."""

    # Run download_data(date_str) in a loop between two dates
    all_dates_str, all_dates = list_of_date_strs(start_date_str, end_date_str)
    for date_str in all_dates_str: download_data(date_str)
    print("All .hdf files downloaded.")

    # Create a list of all the files (which should already exist if you've run download_sequence_of_data())
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepaths = [dirpath + '/concentration_data/' + 'asi-AMSR2-s6250-' + date_str + '-v5.4.hdf' for date_str in all_dates_str]

    # Open the grid and mask files 
    landmask_Ant_fp, landmask_Arc_fp = dirpath + '/landmask_Ant_6.25km.hdf', dirpath + '/landmask_Arc_6.25km.hdf'
    lonLat_Ant_fp, lonLat_Arc_fp = dirpath + '/LongitudeLatitudeGrid-s6250-Antarctic.hdf', dirpath + '/LongitudeLatitudeGrid-n6250-Arctic.hdf'
    landmask_Ant_hdf, landmask_Arc_hdf = SD(landmask_Ant_fp, SDC.READ), SD(landmask_Arc_fp, SDC.READ)
    lonLat_Ant_hdf, lonLat_Arc_hdf = SD(lonLat_Ant_fp, SDC.READ), SD(lonLat_Arc_fp, SDC.READ)
    landmask_Ant_data, landmask_Arc_data = landmask_Ant_hdf.select('landmask Ant 6.25 km').get(), landmask_Arc_hdf.select('landmask Arc 6.25 km').get()
    lon_Ant_data, lon_Arc_data = lonLat_Ant_hdf.select('Longitudes').get(), lonLat_Arc_hdf.select('Longitudes').get()
    lat_Ant_data, lat_Arc_data = lonLat_Ant_hdf.select('Latitudes').get(), lonLat_Arc_hdf.select('Latitudes').get()
    lonLat_Ant_hdf.end(), lonLat_Arc_hdf.end(), landmask_Ant_hdf.end(), landmask_Arc_hdf.end()

    # Init a dataset with the coordinates but no variables
    ds = xr.Dataset(
        data_vars=dict(),
        coords=dict(
            lon=(['x', 'y'], lon_Ant_data),
            lat=(['x', 'y'], lat_Ant_data),
            mask = (['x', 'y'], landmask_Ant_data)),
        attrs={'Description': 'Ice concentration in Antarctic from the AWI sea ice portal',
               'History': "Created by Rowan Brown, 21.01.2025",
               'URL:': 'https://data.meereisportal.de/relaunch/concentration?lang=en'})

    # Loop through the .hdf files, create a dataset, and combine it with ds
    for n,fp in enumerate(filepaths):
        ice_hdf = SD(fp, SDC.READ)
        ice_data = ice_hdf.select('ASI Ice Concentration').get()
        ice_hdf.end()
        ice_ds = xr.Dataset(
            data_vars=dict(ice_conc = (['x', 'y'], ice_data)),
            coords=dict(
                lon=(['x', 'y'], lon_Ant_data),
                lat=(['x', 'y'], lat_Ant_data),
                date=('date', [all_dates[n]])))
        try: # Concat won't work for the first .hdf
            ds = xr.concat([ds, ice_ds], dim='date')
        except: # ...but merge will
            ds = xr.merge([ds, ice_ds])
        print(fp + ' added to .nc')

    ds.to_netcdf(dirpath + '/sea_ice_concentration.nc')
    print(ds)
    print('Ice concentration saved as .nc')

def map_of_ice_conc(date_str):
    
    # Create a list of all the files (which should already exist if you've run download_sequence_of_data())
    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/sea_ice_concentration.nc'
    ds = xr.open_dataset(filepath).sel(date=datetime.strptime(date_str, '%Y%m%d'))

    plt.rcParams["font.family"] = "serif" # change the base font
    f = plt.figure(figsize=(4,2.5)) #2,1,figsize=(4, 6),subplot_kw=dict(projection=projection)) 
    gs = gridspec.GridSpec(1, 1)#, height_ratios=[1, 2])
    southLat, northLat, westLon, eastLon = -85, -55, 179, -179
    land_50m = feature.NaturalEarthFeature('physical', 'land', '50m',edgecolor='black', facecolor='black')
    projection = ccrs.PlateCarree(central_longitude=(westLon+eastLon)/2)#ccrs.AlbersEqualArea(central_longitude=(westLon+eastLon)/2, central_latitude=(southLat+northLat)/2,standard_parallels=(southLat,northLat))
    subplot_kw=dict(projection=projection,aspect=4)
    ax1 = f.add_subplot(gs[0], **subplot_kw)
    ax1.set_extent([westLon, eastLon, southLat, northLat], crs=ccrs.PlateCarree())
    ax1.add_feature(land_50m, color=[0.8, 0.8, 0.8])
    ax1.coastlines(resolution='50m')
    gl = ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, linewidth=0.5)
    gl.top_labels=False #suppress top labels
    gl.right_labels=False #suppress right labels
    gl.rotate_labels=False
    gl.ylocator = mticker.FixedLocator([-85, -80, -75, -70, -65, -60, -55])  #[50, 55, 60, 65, 70, 75, 80])
    gl.xlocator = mticker.FixedLocator([-180,-150,-120,-90,-60,-30,0,30,60,90,120,150,180])
    gl.xlabel_style = {'size': 9}
    gl.ylabel_style = {'size': 9}
    ax1.set_title(date_str[6:]+'.'+date_str[4:6]+'.'+date_str[:4], fontsize=12)
    da = ds['ice_conc'].where(ds['ice_conc']!=0).where(~np.isnan(ds['mask']))
    c = ax1.pcolormesh(ds['lon'],ds['lat'],
                     da,
                     cmap='Blues')#,levels=10)
    cbar = f.colorbar(c, ax=ax1, cmap='Blues', orientation='horizontal')
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_xlabel('Sea ice concentration ($\%$)')

    plt.savefig('Figures/Sea_ice_concentration/Sea_ice_concentration_'+date_str+'.png',dpi=300,bbox_inches='tight')
    print("Ice concentration map saved for date " + date_str)
    plt.close()

if __name__=="__main__":
    sea_ice_conc_nc('20210326', '20220501')
    #all_dates_str, all_dates = list_of_date_strs('20210425', '20220501')
    #for date_str in all_dates_str: map_of_ice_conc(date_str)
