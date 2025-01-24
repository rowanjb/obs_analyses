# Rowan Brown, 23.12.2024
# Munich, Germany
# This file is meant to be a collection of functions that examine the stratification and surface fluxes at the Weddell Sea mooring and elsewhere 

import xarray as xr 
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 
from sea_ice_concentration import select_nearest_coord
from mooring_analyses import open_mooring_ml_data

def open_ERA5_data():
    """Opens ERA5 data from the vicinity of the Weddell Sea mooring."""

    # Open the ERA5 data
    with open('../filepaths/ERA5') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    ds = xr.open_mfdataset(dirpath + '/ERA5_mooring/*.nc')

    #Now you just need to calculate heat fluxes using eg bulk formulae by Large and Yearger. Easy peasy...    
    
    print(ds)
    quit()

def sea_ice():
    """Somehow estiamtes sea ice growth/melt and therefore salt rejection?"""

    # You don't have thickness, so you need to look at 

def WOA_stratification():
    """Somehow calculates measures of stratification within the water column at a chosen location in the WOA data."""

    with open('../filepaths/woa_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    ds_woa_2015 = xr.open_dataset(dirpath + '/WOA_monthly_'+'t'+'_'+str(2015)+'.nc',decode_times=False)
    ds_woa_2015 = ds_woa_2015.rename({'time':'month'}) 
    ds_woa_2015['month'] = ds_woa_2015['month'] - 35.5 # months are saved as arbitrary integers 
    ds_woa_2015['month'] = ds_woa_2015['month'].astype(int) # convert to int 
    #ds_woa_2015 = ds_woa_2015.reindex({'month': ds_mooring['month']}) # reindex means instead of 1-12 we have 4-3

    print(ds_woa_2015)

if __name__=="__main__":
    #open_ERA5_data()
    ds = open_mooring_ml_data()
    #WOA_stratification()
