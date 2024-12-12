# Rowan Brown, 12.12.2024
# Munich, Germany

import scipy.io as spio
import h5py
import numpy as np
import xarray as xr 


def open_mat_final(itp):
    """For opening the level 3 pressure-bin-averaged data at 1-db vertical resolution (i.e., the 'final' files).
    According to: https://www2.whoi.edu/site/itp/data/data-products/"""
    with open('../itp_filepath') as f: dirpath = f.readlines()[0][:-1]  # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath+str(itp)+'/itp'+str(itp)+'final.mat'             # filepath to the processed data mat file
    mat = spio.loadmat(filepath)
    print(mat)
    quit()
    date = mat['date']  # profile start date and time [year month day hour minute second]
    di = mat['di']      # di 1-m bin centers
    lat = mat['lat']    # lat start latitude (N+) of profiles
    lon = mat['lon']    # lon start longitude (E+) of profiles
    p = mat['P']        # P 1-m averaged pressure (dbar)
    s = mat['S']        # S 1-m averaged salinity
    t = mat['T']        # T 1-m averaged temperature (°C)
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
    # ENDED UP REALISING THIS WASN'T USEFUL BECAUSE SOME OF THE .MAT FILES DON'T HAVE SALINITY FILEDS :(

def open_mat_loc(itp):
    """For opening the location data of the ITP.
    I /assume/ this isn't necessary because we can use the lats and lons from the final.mat file.
    But I'll leave this here in case it ends up being useful."""
    with open('../itp_filepath') as f: dirpath = f.readlines()[0][:-1]  # the [0] accesses the first line, and the [:-1] removes the newline tag
    filepath = dirpath+str(itp)+'/itp'+str(itp)+'loc.mat'               # filepath to the location data mat file
    with h5py.File(filepath, 'r') as mat:
        lons = np.array(mat['LON'])
    print(len(lons[0]))
    # ENDED UP REALISING THIS ISN'T USEFUL

if __name__=="__main__":
    itp = 132
    open_mat_final(itp)
