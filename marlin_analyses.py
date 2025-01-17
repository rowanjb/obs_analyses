# Rowan Brown, 13.01.2025
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
    f.supylabel('Depth ($m$)',fontsize=11)
    f.tight_layout()
    plt.savefig('Figures/Marlin_temperature_hovm_8x6.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Marlin_temperature_hovm_8x6.pdf',format='pdf',bbox_inches='tight')

# plotting temperature
def temp_hovm_one_float(ds,id):
    """Created a Hovmöller plot of temperature for just one float with a map."""

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
    ax1.scatter(ds['longitude'].sel(buoy=id).values,ds['latitude'].sel(buoy=id).values,marker='.',s=5,facecolors='red', edgecolors='red')
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
    cax = ds['temperature'].sel(buoy=id).plot.contourf('time','depth',ax=ax2,levels=10,add_colorbar=False)
    cbar = f.colorbar(cax, ax=ax2, cmap=cmap)
    cbar.ax.tick_params(labelsize=9)
    cbar.ax.set_ylabel('Temperature ($\degree C$)')
    ax2.grid(True) 
    ax2.set_ylabel('Depth ($m$)',fontsize=11)
    ax2.set_xlabel('',fontsize=9)
    ax2.set_title('Temperature',fontsize=11)
    ax2.tick_params(axis='both',labelsize=9)
    ax2.invert_yaxis()

    f.suptitle('Float '+str(id),fontsize=12)    
    f.tight_layout()
    plt.savefig('Figures/Marlin_temperature_hovm_4x3_'+str(id)+'.png',bbox_inches='tight',dpi=450)
    plt.savefig('Figures/Marlin_temperature_hovm_4x3_'+str(id)+'.pdf',format='pdf',bbox_inches='tight')

if __name__=="__main__":
    ds = open_marlin_data()
    temp_hovm(ds)
    temp_hovm_one_float(ds,226781)
    temp_hovm_one_float(ds,226782)
    temp_hovm_one_float(ds,226783)
    temp_hovm_one_float(ds,226784)
    temp_hovm_one_float(ds,226785)
    temp_hovm_one_float(ds,226786)
    temp_hovm_one_float(ds,227128)
    temp_hovm_one_float(ds,227129)