# Rowan Brown, 13.12.2024
# Munich, Germany

import xarray as xr 
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 

import time

def get_woa(season:str,decade:int):
    """
    Accesses the T and S seasonal climatology data from WOA.
    Saves as .nc in the group WOA directory.

    Parameters:  
        season (str):       'winter', 'spring', 'summer', or 'autumn'
        dataset (int):      2015, 2005, 1995
    """
    
    season_dict = {'winter': '13', 'spring': '14', 'summer': '15', 'autumn': '16'}
    decade_dict = {2015: 'B5C2', 2005: 'A5B4', 1995: '95A4'}

    url_t = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/temperature/netcdf/"+decade_dict[decade]+"/0.25/woa23_"+decade_dict[decade]+"_t"+season_dict[season]+"_04.nc?crs,lat[0:1:719],lat_bnds[0:1:719][0:1:1],lon[0:1:1439],lon_bnds[0:1:1439][0:1:1],depth[0:1:101],depth_bnds[0:1:101][0:1:1],time[0:1:0],climatology_bounds[0:1:0][0:1:1],t_an[0:1:0][0:1:0][0:1:0][0:1:0],t_mn[0:1:0][0:1:0][0:1:0][0:1:0],t_dd[0:1:0][0:1:0][0:1:0][0:1:0],t_sd[0:1:0][0:1:0][0:1:0][0:1:0],t_se[0:1:0][0:1:0][0:1:0][0:1:0],t_oa[0:1:0][0:1:0][0:1:0][0:1:0],t_ma[0:1:0][0:1:0][0:1:0][0:1:0],t_gp[0:1:0][0:1:0][0:1:0][0:1:0],t_sdo[0:1:0][0:1:0][0:1:0][0:1:0],t_sea[0:1:0][0:1:0][0:1:0][0:1:0]"
    url_s = "https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/salinity/netcdf/"+decade_dict[decade]+"/0.25/woa23_"+decade_dict[decade]+"_s"+season_dict[season]+"_04.nc?crs,lat[0:1:719],lat_bnds[0:1:719][0:1:1],lon[0:1:1439],lon_bnds[0:1:1439][0:1:1],depth[0:1:101],depth_bnds[0:1:101][0:1:1],time[0:1:0],climatology_bounds[0:1:0][0:1:1],s_an[0:1:0][0:1:0][0:1:0][0:1:0],s_mn[0:1:0][0:1:0][0:1:0][0:1:0],s_dd[0:1:0][0:1:0][0:1:0][0:1:0],s_sd[0:1:0][0:1:0][0:1:0][0:1:0],s_se[0:1:0][0:1:0][0:1:0][0:1:0],s_oa[0:1:0][0:1:0][0:1:0][0:1:0],s_ma[0:1:0][0:1:0][0:1:0][0:1:0],s_gp[0:1:0][0:1:0][0:1:0][0:1:0],s_sdo[0:1:0][0:1:0][0:1:0][0:1:0],s_sea[0:1:0][0:1:0][0:1:0][0:1:0]"
    remote_data_t = xr.open_dataset(url_t,decode_times=False)
    remote_data_s = xr.open_dataset(url_s,decode_times=False)
    
    print(remote_data_t.time.to_numpy())
    print(remote_data_s.to_numpy())

    #with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    #filepath = dirpath + '/' + 'seasonal_climatology_' + var + '_' year + season

if __name__=="__main__":
    t = time.time()
    get_woa('winter',1995)
    quit()
    get_woa('spring',1995)
    get_woa('summer',1995)
    get_woa('autumn',1995)
    print(time.time()-t)
