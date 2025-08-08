# Rowan Brown, 13.12.2024
# Munich, Germany
# Switching from functional to OOP with the help of copilot and 
# other web sources, 08.2025

import xarray as xr 
import pandas as pd
import numpy as np
import gsw
from datetime import datetime, timedelta
import scipy.io as spio
import matplotlib.pyplot as plt 

class mooring_ds(xr.Dataset):
    """
    A subclass for datasets of the Weddell Sea mooring.
    (Created partially as an excuse to practice making a class.)
    
    ...
    Methods
    --------
    correct_mooring_salinities()
        corrects salinities using WOA climatologies
    convert_to_daily()
        converts the time delta to daily from the defaults 2 hours
    add_gsw_vars()
        adds gsw variables like SA, pt, etc.

    Note to self: add ability to use WOA or CTD casts for filling and 
    correcting
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
    
    def convert_to_daily(self):
        """Resamples the 2-hourly data to daily"""
        # Obviously this is very simple but I'm always forgetting the 
        # .resample syntax
        self = self.resample(time='D').mean()
        
    def add_gsw_vars(self, pref=0):
        """Adds gsw variables like absolute salinity (SA), conservative
        temperature (CT), pressure calculated from nominal depth (P), and 
        potential temperature (pt). 
        
        Parameters
        --------
            pref : int, default 0
                reference pressure used in calculating potential density
        """
        lon, lat = -27.0048, -69.0005
        self['p_from_z'] = gsw.p_from_z(self['depth'],lat)
        self['SA'] = gsw.SA_from_SP(self['S'],self['p_from_z'],lon,lat)
        self['CT'] = gsw.CT_from_t(self['SA'],self['T'],self['p_from_z'])
        self['pt'] = gsw.pt_from_t(self['SA'],self['T'],self['p_from_z'],pref)
        self['pt'].attrs['reference pressure [dbar]'] = pref

def open_mooring_profiles_data():
    """
    Opens CTD data from profiles taken during the mooring launch/pickup cruises.
    Work in progress...
    Use this to "correct" instead of WOA?
    """

    print("Beginning to open the CTD data")
    
    # This function requires the matlab engine package, installed using
    # python -m pip install matlabengine==9.13.1
    # Requires MATLAB/R2022b and gcc/12.1.0 to be loaded
    try:
        import matlab.engine
        eng = matlab.engine.start_matlab()
    except:
        print("Failed to import matlab.engine")
        print("Check that you have loaded the following environment modules:")
        print("MATLAB/R2022b")
        print("gcc/12.1.0")
        quit()

    # File path is saved in a text file to avoid publishing it to GitHub
    with open('../filepaths/mooring_filepath') as f: 
        dirpath = f.readlines()[0][:-1] 
    
    # Creating the full filepaths to the .mat files
    filepaths = {
        '124': dirpath + '/CTD/Profiles/117_1.mat',
        '129': dirpath + '/CTD/Profiles/dPS129_072_01.mat',
    }

    #== PS124 ==#
    # Need different approaches for each cruise because diff data struct
    # fieldnames_124 = eng.fieldnames(CTD_124) # Example usage
    eng.workspace['CTD_124'] = eng.load(filepaths['124'])
    CTD_124 = eng.workspace['CTD_124']['S']
    unpack_var = lambda var : [i[0] for i in eng.getfield(CTD_124, var)]
    ds_124 = xr.Dataset(
        data_vars=dict(
            T=(["P"], unpack_var('TEMP'),{'units':'deg C'}),
            S=(["P"], unpack_var('SALT'),{'units':'PSU'}),
        ),
        coords=dict(
            P=(["P"], unpack_var('PRES'),{'units':'dbar'}),
            datetime=pd.to_datetime(eng.getfield(CTD_124, 'DATETIME')),
            lon=(eng.getfield(CTD_124, 'LON')),
            lat=(eng.getfield(CTD_124, 'LAT')),
        ),
        attrs=dict(
            description="CTD data for cruise PS124",
            institute=(eng.getfield(CTD_124,'INSTITUTE')),
            institute_country=(eng.getfield(CTD_124,'INST_COUNTRY')),
            instrument=(eng.getfield(CTD_124,'INSTRUMENT')),
            instrument_serial_number=(eng.getfield(CTD_124,'SN')),
            ship=(eng.getfield(CTD_124,'Ship')),
            cruise=(eng.getfield(CTD_124,'CRUISE')),
            station=(eng.getfield(CTD_124,'STATION ')),
            water_depth=(str(eng.getfield(CTD_124,'WATERDEPTH'))),
            ),
    )

    #== PS129 ==#
    #print(eng.fieldnames(md_129))
    eng.workspace['CTD_129'] = eng.load(filepaths['129'])
    md_129 = eng.workspace['CTD_129']['HDR']
    CTD_129 = eng.workspace['CTD_129']['DATA']    
    CTD_129 = eng.table2struct(CTD_129,"ToScalar",True) # Simplicity
    unpack_var = lambda var : [i[0] for i in eng.getfield(CTD_129, var)]
    ds_129 = xr.Dataset(
        data_vars=dict(
            T=(
                ["sensor","P"], 
                [unpack_var('TEMP0'),unpack_var('TEMP1')],
                {'units':'deg C'}
            ),
            S=(
                ["sensor","P"], 
                [unpack_var('SALP0'),unpack_var('SALP1')],
                {'units':'PSU'}
            ),
        ),
        coords=dict(
            sensor=([0,1]),
            P=(["P"], unpack_var('PRES'),{'units':'dbar'}),
            datetime=pd.to_datetime(eng.getfield(md_129, 'DATETIME')),
            lon=(eng.getfield(md_129, 'LON')),
            lat=(eng.getfield(md_129, 'LAT')),
        ),
        attrs=dict(
            description="CTD data for cruise PS129",
            institute=(eng.getfield(md_129,'INSTITUTE')),
            institute_country=(eng.getfield(md_129,'INST_COUNTRY')),
            instrument=(eng.getfield(md_129,'INSTRUMENT')),
            instrument_serial_number=(eng.getfield(md_129,'SN')),
            ship=(eng.getfield(md_129,'SHIP')),
            cruise=(eng.getfield(md_129,'CRUISE')),
            station=(eng.getfield(md_129,'STATION ')),
            water_depth=(str(eng.getfield(md_129,'WDEPTH'))),
            ),
    )

    #== Combining ==# 
    # Now that the datasets are created, we can combine them 
    def make_vars(var):
        var = [
            ds_124[var].interp(P=ds_129['P']).values,
            ds_129[var].mean(dim='sensor').values
        ]   
        return var 
    def make_coords(coord):
        return [ds_124[coord].values, ds_124[coord].values]
    def make_attrs(attr):
        return 'PS124: '+ds_124.attrs[attr]+' PS129: '+ds_129.attrs[attr]
    ds = xr.Dataset(
        data_vars=dict(
            T = (["datetime", "P"], make_vars('T')),
            S = (["datetime", "P"], make_vars('S')),
        ),
        coords=dict(
            P = (["P"], ds_129['P'].values),
            datetime = (["datetime"], make_coords('datetime')),
            lon = (["datetime"], make_coords('lon')),
            lat = (["datetime"], make_coords('lat')),
        ),
        attrs=dict(
            description="CTD data for cruises PS124 and PS129",
            institute=make_attrs('institute'),
            institute_country=make_attrs('institute_country'),
            instrument=make_attrs('instrument'),
            instrument_serial_number=make_attrs('instrument_serial_number'),
            ship=make_attrs('ship'),
            cruise=make_attrs('cruise'),
            station=make_attrs('station'),
            water_depth=make_attrs('water_depth'),
            ),
    )

    print("Completed opening the CTD data")
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
    #ds.fill_mooring()
    #ds.convert_to_daily()
    #ds.add_gsw_vars()
    #ds = open_mooring_profiles_data()