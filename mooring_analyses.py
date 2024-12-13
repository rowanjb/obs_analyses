# Rowan Brown, 13.12.2024
# Munich, Germany

import xarray as xr 
import pandas as pd
from datetime import datetime, timedelta
import scipy.io as spio

def open_mooring_data():
    """Opens data from the Weddell Sea mooring."""

    with open('../filepaths/mooring_filepath') as f: dirpath = f.readlines()[0][:-1] # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath + '/CTD/dPS129_072_01.mat'
    mat = spio.loadmat(filepath)
    print(mat)
    quit()


    dirpath+str(itp)+'/itp'+str(itp)+'grddata/itp'+str(itp)+'grd0694.dat'             # filepath to the processed data mat file
    locpd = pd.read_csv(filepath, sep='\s+', nrows=2, engine='python')
    datapd = pd.read_csv(filepath, sep='\s+', skiprows=2, skipfooter=1, engine='python')
    lon = locpd.iloc[0,2]
    lat = locpd.iloc[0,3]
    pressure = datapd['pressure(dbar)']
    temperature = datapd['temperature(C)']
    salinity = datapd['salinity']
    year = datetime(datapd['%year'],12,31)
    print('test')
    quit()
    day = datapd['day']
    date = datetime(year.iloc[10],12,31) + timedelta(days=1) #timedelta(days=day[10])
    print(date)
    
    quit()
    ds = xr.Dataset(
            data_vars=dict(
                temperature=(['loc', 'di', 'date'], t),
                pressure=(['loc', 'di', 'date'], p),
                salinity=(['loc', 'di', 'date'], s),
            ),
            coords=dict(
                lon=('loc',lon),
                lat=('loc',lat),
                di=di,
                date=date,
            ),
            attrs=dict(description='Test desc.'),
        )
    print(ds)
    '''
    
if __name__=="__main__":
    open_mooring_data()
