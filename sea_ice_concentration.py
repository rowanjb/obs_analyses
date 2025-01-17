# Rowan Brown, 17.01.2024
# Munich, Germany

import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt
from pyhdf.SD import SD, SDC
import subprocess
import os

def download_data(date_str, output):
    """Credit for this function and the Bash script that it calls goes to copilot.
    They work together to download hdf (version 4, I think) files of daily sea ice concentration in Antarctica.
    """

    # Call the bash script with the date parameter
    result = subprocess.run(['./download_conc_data.sh', date_str, output], capture_output=True, text=True)

    # Print the output and return code
    print(result.stdout)
    return result.returncode

def get_sea_ice_concentration(date):

    with open('../filepaths/sea_ice_concentration') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag

    filename = dirpath + '/concentration_data/' + 'asi-AMSR2-s6250-' + date + '-v5.4.hdf'
    
    if os.path.isfile(filename)==False:
        return_code = download_data(date,filename)
        if return_code == 0:
            print("Script executed successfully")
        else:
            print("Script execution failed")

    hfile = SD(filename, SDC.READ)
    ds = hfile.select('ASI Ice Concentration')
    data = ds.get()
    hfile.end()



if __name__=="__main__":
    get_sea_ice_concentration(date='20250104')
