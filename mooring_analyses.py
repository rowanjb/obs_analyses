# Rowan Brown, 13.12.2024
# Munich, Germany
# Switching from functional to OOP with the help of copilot and 
# other web sources, 08.2025

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
from typing import Literal

import sys
sys.path.insert(1, '../model_analyses/')
import cell_thickness_calculator as ctc

class mooring_ds(xr.Dataset):
    """
    A subclass for datasets of the Weddell Sea mooring.
    (Created partially as an excuse to practice making a class.)
    
    ...
    Methods
    --------
    correct_mooring_salinities()
        corrects salinities using WOA climatologies
    fill_mooring()
        populates the datasets in depth using WOA climatoloties

    Note to self: add ability to use WOA or CTD casts for filling and 
    correcting. Also add convert to daily and add SA, PT, etc. methods
    """
    __slots__ = () # Note to self: Add attributes here
    def correct_mooring_salinities(self):
        """
        The salinities from the lower two sensors seem, basically, wrong.
        Here, I'm equating the sensors' means with those of WOA climatologies.
        In the future, I might at functionality to correct using the CTD casts.
        
        For info on WOA data, see:
        WORLD OCEAN ATLAS 2023
        Product Documentation
        """

        print("Beginning to correct the mooring data")
        
        # Calculate the mean salinites at the two "bad sensors"
        S = self['S'] # Extract as a dataarray for easy handling
        S_srfce_mean_mooring = S.sel(depth=-50).mean(dim='time').values
        S_upper_mean_mooring = S.sel(depth=-125).mean(dim='time').values
        S_lower_mean_mooring = S.sel(depth=-220).mean(dim='time').values

        # Open the WOA data and do some light processes
        with open('../filepaths/woa_filepath') as f: 
            dirpath = f.readlines()[0][:-1] 
        ds_woa = xr.open_dataset(
            dirpath+'/WOA_monthly_'+'s'+'_'+str(2015)+'.nc',
            decode_times=False
        )
        ds_woa = ds_woa.rename({'time':'month'}) 
        ds_woa['month'] = ds_woa['month'] - 35.5 
        # (Months in WOA are saved as arbitrary reals)

        # Calculate the yearly average, using weighting procedure from: 
        # xarray example "area_weighted_temperature"
        # Note the s_an are practical salinities
        ds_woa['weights'] = ('month',[31,28,31,30,31,30,31,31,30,31,30,31]) 
        ds_woa_weighted = ds_woa['s_an'].weighted(ds_woa['weights'])
        woa_weighted_mean = ds_woa_weighted.mean('month')

        S_srfce_mean_woa = woa_weighted_mean.interp(depth=50).values
        S_upper_mean_woa = woa_weighted_mean.interp(depth=125).values
        S_lower_mean_woa = woa_weighted_mean.interp(depth=220).values

        S_srfce_mean_anomaly = S_srfce_mean_mooring-S_srfce_mean_woa
        S_upper_mean_anomaly = S_upper_mean_mooring-S_upper_mean_woa
        S_lower_mean_anomaly = S_lower_mean_mooring-S_lower_mean_woa

        S = xr.where(S['depth']==-50,S.sel(depth=-50)-S_srfce_mean_anomaly,S)
        S = xr.where(S['depth']==-125,S.sel(depth=-125)-S_upper_mean_anomaly,S)
        S = xr.where(S['depth']==-220,S.sel(depth=-220)-S_lower_mean_anomaly,S)
        
        # Reassign the corrected values
        self['S'] = S.transpose('time','depth')
        
        print("Mooring data corrected")

    def fill_mooring(self):
        """
        For filling in the mooring dataset S and T by adding 
        depth levels. Currently uses WOA climatology data but in the 
        future might I start using CTD casts.
        Note season is relative to the N.H., so the default is 
        autumn (i.e., Southern Hemisphere spring).
        
        For info on WOA data, see:
        WORLD OCEAN ATLAS 2023
        Product Documentation
        """

        print("Beginning to fill the mooring data in the vertical")

        # Calculating thickness levels
        # You can change some of these parameters if desired
        # But these are the values that I've used in the model
        depth = 500 # Model grid depth
        num_levels = 10 #50 # Grid size
        x1, x2 = 1, num_levels # Indices of top and bottom cells
        fx1 = 1 # Depth of bottom of top cell
        min_slope = 1 # Minimum slope (should probably > x1)
        A, B, C, _, _ = ctc.find_parameters(x1,x2,fx1,depth,min_slope)
        dz = ctc.return_cell_thicknesses(x1,x2,depth,A,B,C) 

        # Depths used in the model (calc'd to the centre of the cells)
        z = np.zeros(len(dz))
        for i,n in enumerate(dz):
            if i==0: z[i] = n/2
            else: z[i] = np.sum(dz[:i]) + n/2

        # Opening the WOA data (seasons are ref'd to the N.H.
        # e.g., ['winter', 'spring', 'summer', 'autumn'] 
        with open('../filepaths/woa_filepath') as f: 
            dirpath = f.readlines()[0][:-1] # 
        das = xr.open_dataset(
            dirpath + '/WOA_monthly_'+'s'+'_'+str(2015)+'.nc',
            decode_times=False
        )
        das = das['s_an'] # s_an is practical salinity
        dat = xr.open_dataset(
            dirpath + '/WOA_monthly_'+'t'+'_'+str(2015)+'.nc',
            decode_times=False
        )
        dat = dat['t_an'] # in situ, see: https://catalog.data.gov...
        # /dataset/world-ocean-atlas-2023?utm_source=chatgpt.com
        woa_month_dict = {'August': 8, 'September': 9, 'October': 10}
        s_woa = das.isel(time=woa_month_dict['September']).interp(depth=z)
        t_woa = dat.isel(time=woa_month_dict['September']).interp(depth=z) 
        
        self["z"] = z
        empty_filled_arr = np.empty((len(self['time']),len(z)))
        self["S_filled"] = (("time", "z"), empty_filled_arr)
        self["T_filled"] = (("time", "z"), empty_filled_arr)
        self["S_woa"] = (("z"), s_woa.data)
        self["T_woa"] = (("z"), t_woa.data)
        self["S_shift"] = self["S_woa"].interp(z=10.111) - self["S"]
        self["T_shift"] = self["T_woa"].interp(z=10.111) - self["T"]
        
        print(self)
        quit()

        for n,d in enumerate(z):
            
            if d<50: 
                mean_diff_s = dss[0] - s_woa[id50]
                mean_diff_t = dst[0] - t_woa[id50]
                s[n] = s_woa[n] + mean_diff_s
                t[n] = t_woa[n] + mean_diff_t
            elif d<125:
                del_s = dss[1] - dss[0]
                del_t = dst[1] - dst[0]
                weight = (d-50)/(125-50)
                s[n] = dss[0] + del_s*weight
                t[n] = dst[0] + del_t*weight
            elif d<220:
                del_s = dss[2] - dss[1]
                del_t = dst[2] - dst[1]
                weight = (d-125)/(220-125)
                s[n] = dss[1] + del_s*weight
                t[n] = dst[1] + del_t*weight
            else:
                mean_diff_s = dss[2] - s_woa[id220]
                mean_diff_t = dst[2] - t_woa[id220]
                s[n] = s_woa[n] + mean_diff_s
                t[n] = t_woa[n] + mean_diff_t
            
def open_mooring_profiles_data():
    """Opens CTD data from profiles taken during the mooring launch/pickup cruises.
    Work in progress...
    Use this to "correct" instead of WOA?
    """

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
            time=new_datetime_coords,
            depth=[-50,-90,-125,-170,-220,-250],
        ),
        attrs=dict(description="Mooring data..."),
    )

    #THIS SHOULD USE SA NOT S
    #ds = ds.assign_coords(p_from_z=gsw.p_from_z(ds['depth'],-69.0005))
    #ds['pot_rho'] = gsw.pot_rho_t_exact(ds['S'],ds['T'],ds['p_from_z'],0) - 1000
    
    return ds


def open_mooring_data():
    """Opens the mooring .mat file(s) and converts into a mooring_ds 
    object"""

    print("Beginning to open the mooring data")

    # Open file path (saved in txt to avoid publishing to GitHub)
    # [0] accesses the first line, and [:-1] removes the newline tag
    with open('../filepaths/mooring_filepath') as f: 
        dirpath = f.readlines()[0][:-1] 

    # Creating the full filepaths to the .mat files
    # BGC_SBE is the main data, and sal_BGC has corrected salinities
    # for two of the sensors
    filepath_BGC_SBE = dirpath + '/CTD/Mooring/BGC_SBE.mat' 
    filepath_sal_BGC = dirpath + '/CTD/Mooring/sal_BGC.mat' 

    mat = spio.loadmat(filepath_BGC_SBE)['SBE'] # SBE = Sea Bird
    mat_corr = spio.loadmat(filepath_sal_BGC) 

    # Extracting the needed data from the main .mat file
    # Note 'P' is missing data at all but the -50 and -125 sensors
    jul = mat['jul'][0]
    T, S, P = mat['T'][0], mat['S'][0], mat['P'][0] 

    # Extracting the corrected salinities and updating the S array
    Sal_449 = mat_corr['Sal_449']
    Sal_2100 = mat_corr['Sal_2100'] 
    S[4] = Sal_449
    S[2] = Sal_2100
    
    # For getting the variable at dt intervals
    start_date = datetime(2020,12,31,0,0,0) # Days are from 31/12
    def daily_avg_mooring_data(inmat): 

        # Looping through each depth level, 0 to 5
        dfs = []
        for sensor_id in range(len(jul)): 
            
            # Put data into a pandas dataframe
            dates = pd.to_datetime( 
                [start_date+timedelta(days=i[0]) for i in jul[sensor_id]]
            ) 
            
            # If a non-empty list (sometimes data is missing)
            if len(inmat[sensor_id]) > 0: 
                var_data = [i[0] for i in inmat[sensor_id]]
            else:
                var_data = [np.nan for i in dates]
            df = pd.DataFrame(
                data={'dates': dates, sensor_id: var_data}
            )
            df = df.set_index('dates') 
            df = df[
                (df.index>'2021-04-01 00:00:00') &
                #(df.index<'2022-04-01 01:00:00')
                (df.index<'2021-04-10 01:00:00')
            ]

            # Use the very nice pandas resample method
            df = df.resample('h').mean()

            dfs.append(df)

        return pd.concat(dfs,axis=1)
    
    T_resampled = daily_avg_mooring_data(T)
    S_resampled = daily_avg_mooring_data(S)
    P_resampled = daily_avg_mooring_data(P)

    ds = mooring_ds(
        data_vars=dict(
            T=(["time","depth"], T_resampled.to_numpy()),
            S=(["time","depth"], S_resampled.to_numpy()),
            P=(["time","depth"], P_resampled.to_numpy()),
        ),
        coords=dict(
            time=T_resampled.index.to_numpy(),
            depth=[-50,-90,-125,-170,-220,-250],
        ),
        attrs=dict(description="Mooring data"),
    )
    
    # The 50 m sensor only has 2-hourly data, so I'm cutting it 
    # down here and changing the key to "2_hours"
    ds = ds.isel(time=slice(0,-1,2)) 
    
    print("Mooring data opened")

    return ds

if __name__=="__main__":   
    
    ds = open_mooring_data()
    #ds.correct_mooring_salinities()
    ds.fill_mooring()