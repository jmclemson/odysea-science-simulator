import numpy as np
import xarray as xr
import glob
import os
import pandas as pd

from odysim import utils
    
    
# class WebGriddedModel:
# TODO: Implement the LLC 4320 model through the xmitgcm llcreader interface
#       or some other cloud readable interface.

class GriddedModel:
    
    """
    
    A class that holds functions for loading and co-locating ocean/atmosphere model data.
    Ocean model data is expected to be gridded in lat/lon with individual files for each
    time step. 
    
    Take this code as a starting point and adapt to your own specific model as needed.
    
    """
    

    def __init__(self,model_folder='/u/bura-m0/hectorg/COAS/llc2160/HighRes/',
                 u_folder='U',v_folder='V',current_fname=None,wind_x_folder='oceTAUX',wind_y_folder='oceTAUY',
                 wind_fname=None,u_varname='U',v_varname='V',wind_x_varname='oceTAUX',wind_y_varname='oceTAUY',
                 variable_selector='winds+currents',wind_var='speed',search_string = '/*.nc',preprocess=None,n_files=-1):

        """
        Initialize a GriddedModel object.
        
        Args:
            model_folder (str): Top-level folder for model data. Contains sub folders for each variable.
            u_folder (str): Sub-folder containing model U current data (East-West currents).
            v_folder (str): Sub-folder containing model V current data (North-South currents).
            current_fname (str): File name of dataset containing combined current data (East-West and North-South currents).
                - Relavant only when using n_files="combined" to read a single dataset with both current variables.
            wind_x_folder (str): Sub-folder containing model U wind data (10m zonal wind speed or East-West wind stress).
            wind_y_folder (str): Sub-folder containing model V wind data (10m meridonal wind speed or North-South wind stress).
            wind_fname (str): File name of dataset containing combined wind data (East-West and North-South wind speed or stress).
                - Relavant only when using n_files="combined" to read a single dataset with both wind variables.
            u_varname (str): Variable name inside model netcdf files for U current.
            v_varname (str): Variable name inside model netcdf files for V current.
            wind_x_varname (str): Variable name inside model netcdf files for U wind data.
            wind_y_varname (str): Variable name inside model netcdf files for V wind data.
            variable_selector (str): String indicating which variables are present in model data.
                - Include sub-string "wind" to load wind model data.
                - Include sub-string "current" to load current model data.
            wind_var (str): String indicating whether 10m wind speeds or wind stress are loaded.
                - Include sub-string "speed" to indicate 10m wind speed data.
                - Include sub-string "stress" to indicate wind stress data.
            search_string (str): File extension for model data files.
            preprocess (function): function to pass to xarray.open_mfdataset for preprocessing.
            n_files (int or str): number of files to load, 0:n_files. Used to reduce load if many files are available in the model folder.
                - Use n_files="combined" to open a single dataset storing multiple variables
            
        Returns:
            GriddedModel obect

        """


        if 'current' in variable_selector:
            if n_files == 'combined':
                file = os.path.join(model_folder, current_fname)

                dataset = xr.open_dataset(file, chunks='auto')
                self.U = dataset[u_varname].to_dataset(name=u_varname)
                self.V = dataset[v_varname].to_dataset(name=v_varname)

            else:
                u_search = os.path.join(model_folder, u_folder)
                v_search = os.path.join(model_folder, v_folder)

                u_files = np.sort(glob.glob(u_search + search_string))[0:n_files]
                v_files = np.sort(glob.glob(v_search + search_string))[0:n_files]

                self.U = xr.open_mfdataset(u_files,parallel=True,preprocess=preprocess)
                self.V = xr.open_mfdataset(v_files,parallel=True,preprocess=preprocess)

            self.u_varname = u_varname
            self.v_varname = v_varname

        if 'wind' in variable_selector:
            if n_files == 'combined':
                file = os.path.join(model_folder, wind_fname)

                dataset = xr.open_dataset(file, chunks='auto')
                self.wind_x = dataset[wind_x_varname].to_dataset(name=wind_x_varname)
                self.wind_y = dataset[wind_y_varname].to_dataset(name=wind_y_varname)
            else:
                wind_x_search = os.path.join(model_folder, wind_x_folder)
                wind_y_search = os.path.join(model_folder, wind_y_folder)
        
                wind_x_files = np.sort(glob.glob(wind_x_search + search_string))[0:n_files]
                wind_y_files = np.sort(glob.glob(wind_y_search + search_string))[0:n_files]

                self.wind_x = xr.open_mfdataset(wind_x_files,parallel=True,preprocess=preprocess)
                self.wind_y = xr.open_mfdataset(wind_y_files,parallel=True,preprocess=preprocess)

            self.wind_x_varname = wind_x_varname
            self.wind_y_varname = wind_y_varname
            self.wind_var_type = wind_var
        
        
    def colocatePoints(self,lats,lons,times):
        
        
        """
        Colocate model data to a set of lat/lon/time query points. 
            Ensure that lat/lon/time points of query exist within the loaded model data.
        
        Args:
            lats (numpy.array): latitudes in degrees
            lons (numpy.array): longitudes in degrees
            times (numpy.array): times represented as np.datetime64

        Returns:
           Model data linearly interpolated to the lat/lon/time query points.
           
           u: colocated model u currents.
           v: colocated model v currents.
           wx: colocated model u winds.
           wy: colocated model v winds.

        """

        
        if len(times) == 0:
            return [],[]
            
        ds_u =  self.U.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')

        ds_v =  self.V.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')
        
        ds_wx =  self.wind_x.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')

        ds_wy =  self.wind_y.interp(time=xr.DataArray(times.flatten(), dims='z'),
                            lat=xr.DataArray(lats.flatten(), dims='z'),
                            lon=xr.DataArray(lons.flatten(), dims='z'),
                            method='linear')
        

        u=np.reshape(ds_u[self.u_varname].values,np.shape(lats))
        v=np.reshape(ds_v[self.v_varname].values,np.shape(lats))
        wx=np.reshape(ds_wx[self.wind_x_varname].values,np.shape(lats))
        wy=np.reshape(ds_wy[self.wind_y_varname].values,np.shape(lats))

        return u,v,wx,wy
        
        
    def colocateSwathCurrents(self,orbit):

        """
        Colocate model current data to a swath (2d continuous array) of lat/lon/time query points. 
            Ensure that lat/lon/time points of query exist within the loaded model data.
        
        Args:
            orbit (object): xarray dataset orbit generated via the orbit.getOrbit() call. 
        Returns:
           original orbit containing model data linearly interpolated to the orbit swath.
                   new data is contained in u_model, v_model

        """
        
        lats  = orbit['lat'].values.flatten()
        lons  = orbit['lon'].values.flatten()
        times = orbit['sample_time'].values.flatten()

        ds_u =  self.U.interp(time=xr.DataArray(times, dims='z'),
                            lat=xr.DataArray(lats, dims='z'),
                            lon=xr.DataArray(lons, dims='z'),
                            method='linear')

        ds_v =  self.V.interp(time=xr.DataArray(times, dims='z'),
                            lat=xr.DataArray(lats, dims='z'),
                            lon=xr.DataArray(lons, dims='z'),
                            method='linear')


        u_interp = np.reshape(ds_u[self.u_varname].values,np.shape(orbit['lat'].values))
        v_interp = np.reshape(ds_v[self.v_varname].values,np.shape(orbit['lat'].values))

        orbit = orbit.assign({'u_model': (['along_track', 'cross_track'], u_interp),
                              'v_model': (['along_track', 'cross_track'], v_interp)})

        return orbit
    
    def colocateSwathWinds(self,orbit):

        """
        Colocate model wind data to a swath (2d continuous array) of lat/lon/time query points. 
            Ensure that lat/lon/time points of query exist within the loaded model data.
        
        Args:
            orbit (object): xarray dataset orbit generated via the orbit.getOrbit() call. 
        Returns:
           original orbit containing model data linearly interpolated to the orbit swath.
                   new data is contained in u10_model, v10_model, wind_speed_model, wind_dir_model, and 
                   in  tx_model, ty_model if wind stress data was passed during initialization.

        """
        
        lats  = orbit['lat'].values.flatten()
        lons  = orbit['lon'].values.flatten()
        times = orbit['sample_time'].values.flatten()

        ds_wx =  self.wind_x.interp(time=xr.DataArray(times, dims='z'),
                            latitude=xr.DataArray(lats, dims='z'),
                            longitude=xr.DataArray(lons, dims='z'),
                            method='linear')

        ds_wy =  self.wind_y.interp(time=xr.DataArray(times, dims='z'),
                            latitude=xr.DataArray(lats, dims='z'),
                            longitude=xr.DataArray(lons, dims='z'),
                            method='linear')


        wx_interp = np.reshape(ds_wx[self.wind_x_varname].values,np.shape(orbit['lat'].values))
        wy_interp = np.reshape(ds_wy[self.wind_y_varname].values,np.shape(orbit['lat'].values))


        if 'stress' in self.wind_var_type:
            wind_speed = utils.stressToWind(np.sqrt(wx_interp**2 + wy_interp**2))
            wind_dir = np.arctan2(wx_interp, wy_interp) # in rad
            u10 = wind_speed * np.sin(wind_dir)
            v10 = wind_speed * np.cos(wind_dir)
            
        else:
            wind_speed = np.sqrt(wx_interp**2 + wy_interp**2)
            wind_dir = np.arctan2(wx_interp, wy_interp) # in rad
            u10 = wx_interp
            v10 = wy_interp

        
        orbit = orbit.assign({'u10_model': (['along_track', 'cross_track'], u10),
                              'v10_model': (['along_track', 'cross_track'], v10)})

        orbit = orbit.assign({'wind_speed_model': (['along_track', 'cross_track'], wind_speed),
                              'wind_dir_model': (['along_track', 'cross_track'], wind_dir*180/np.pi)})
        
        
        if 'stress' in self.wind_var_type:
            orbit = orbit.assign({'tx_model': (['along_track', 'cross_track'], wx_interp),
                                  'ty_model': (['along_track', 'cross_track'], wy_interp)})
            
            orbit['u10_model'].attrs['Units'] = 'Current Relative'
            orbit['v10_model'].attrs['Units'] = 'Current Relative'
            orbit['wind_speed_model'].attrs['Units'] = 'Current Relative'
            orbit['wind_dir_model'].attrs['Units'] = 'Current Relative'
        
        
        return orbit
    
    
def addTimeDim(ds):
    
    """
    Helper function for open_mfdataset. Very specific to a set of model data used at JPL.
        You may need something similar, but probably not exactly this.
        Adds an extra time dimension to a xarray dataset as it is opened so that open_mfdataset
        can stack data along that dimension.
        Looks at the filename from the opened netcdf file to deterimine the time dimension to add.

    Args:
        ds (xarray dataset): dataset that is opened by open_mfdataset.
    Returns:
        ds (xarray dataset): Original dataset with added time dimension.

    """
    
    
    ds = ds.isel(time=0)
    fn = os.path.basename(ds.encoding["source"])
    time_str = fn.split('_')[-1].split('.')[0]
    time = pd.to_datetime(time_str, format='%Y%m%d%H')
    ds = ds.expand_dims(time=[time])

    #display(ds)
    return ds


def addTimeDimCoarse(ds):
    
    """
    Helper function for open_mfdataset. Very specific to a set of model data used at JPL.
        You may need something similar, but probably not exactly this.
        Adds an extra time dimension to a xarray dataset as it is opened so that open_mfdataset
        can stack data along that dimension.
        Looks at the filename from the opened netcdf file to deterimine the time dimension to add.

    Args:
        ds (xarray dataset): dataset that is opened by open_mfdataset.
    Returns:
        ds (xarray dataset): Original dataset with added time dimension.

    """
    
    fn = os.path.basename(ds.encoding["source"])
    time_str = fn.split('.')[0].split('_')[-1]

    time = pd.to_datetime(time_str, format='%Y%m%d%H')
    ds = ds.expand_dims(time=[time])

    return ds
