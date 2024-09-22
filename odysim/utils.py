import numpy as np
import scipy
import os
#from numba import jit
import matplotlib.pyplot as plt
from cartopy import config
import cartopy.crs as ccrs
from scipy import stats
import flox.xarray
import xarray as xr
import datetime
import cf_xarray as cfxr

import os
import importlib.resources as import_resources
from odysim import cartopy_files, swath_sampling, errors, colocate_model

try:
    cf = import_resources.files(cartopy_files)
    os.environ["CARTOPY_USER_BACKGROUNDS"] = cf
except:
    # for some reason, sometimes import_resources retruns a mutliplexedpath instead of a string!
    cf = str(import_resources.files(cartopy_files)).split("'")[0]
    os.environ["CARTOPY_USER_BACKGROUNDS"] = cf



from scipy.interpolate import UnivariateSpline

def gridOrbits(raw_orbit, bin_coordinates, single_time = None):
    '''
    Interpolate ODYSEA orbit data to regular lat-lon or lat-lon-time grid

    Args:
        raw_orbit (xarray.dataset): ODYSEA dataset with "along_track" and "cross_track" as coordinates
            - At minimum must have "lat" and "lon" as variables
        bin_coordinates (xarray.dataset): Dataset containing latitude, longitude, and optionally time coordinates
                                         to grid data onto
        single_time (boolean): If true, regrid data using time bins generated from raw_orbit.sample_time instead
                              of time bins from bin_coordinates

    Returns:
        xarray dataset with variables from raw_orbit and coordinates of "lon", "lat", and optionally "time"
    '''
    
    # Shift lat/lon coordinates by 1/2 their spacing (so that bins are centered on the actual coordinate points)
    lon_shift = (bin_coordinates.lon.values[1] - bin_coordinates.lon.values[0])/2
    lat_shift = (bin_coordinates.lat.values[1] - bin_coordinates.lat.values[0])/2
    # When shift left, must add one coordinate to the right end to define right edge of last bin (to preserve shape through binning)
    lon_coords_shifted = np.append((bin_coordinates.lon.values - lon_shift), (bin_coordinates.lon.values[-1] + lon_shift))
    lat_coords_shifted = np.append((bin_coordinates.lat.values - lon_shift), (bin_coordinates.lat.values[-1] + lat_shift))

    # Create masked lon/lat to filter parameters
    masked_lon = raw_orbit.lon.fillna(-999)
    masked_lat = raw_orbit.lat.fillna(-999)

    # If time coordinates given in bin_coordinates, create shifted time coordinates, then run time-inclussive binning / formatting operations
    if 'time' in bin_coordinates.coords or single_time is not None:

        if single_time is not None:   
            # If using only 1 orbit, custom make 1/2 time bins to guarentee that data fits
            t1 = np.array(raw_orbit.sample_time.min(skipna=True).values, dtype='datetime64[h]')
            t2 = np.array(raw_orbit.sample_time.max(skipna=True).values + np.timedelta64(30, 'm'), dtype='datetime64[h]') + np.timedelta64(1, 'm')
            time_shift = np.timedelta64(1800, 's')
            time_coords = np.arange(t1, t2, np.timedelta64(1,'h'), dtype='datetime64[ns]')
            time_coords_shifted = np.append((time_coords - time_shift), (time_coords[-1] + time_shift))
        else:
            time_coords = bin_coordinates.time.values
            time_shift = (time_coords[1] - time_coords[0])/2
            time_coords_shifted = np.append((time_coords - time_shift), (time_coords[-1] + time_shift))

        masked_time = raw_orbit.sample_time.fillna(np.datetime64('1000-01-01T00:00'))

        # Project orbits data onto new coordinates using an nanmean operation in each bin
        binned_orbit_raw = flox.xarray.xarray_reduce(raw_orbit, masked_lon, masked_lat, masked_time,
                                                     func='nanmean', dim=('along_track', 'cross_track'), fill_value=np.nan,
                                                     expected_groups=(lon_coords_shifted, lat_coords_shifted, time_coords_shifted), isbin=True)
    
        # Re-assign and rename dimension coordinates from pandas interval objects to the grid coordinate points
        # Remove lon and lat variables from dataset
        binned_orbit_1 = binned_orbit_raw.where((binned_orbit_raw.swath_blanking != np.nan) & (binned_orbit_raw.swath_blanking < 1))
        gridded_orbit = binned_orbit_1.assign_coords({'lon_bins':bin_coordinates.lon.values, 'lat_bins':bin_coordinates.lat.values,
                                                      'sample_time_bins':time_coords})
        gridded_orbit = gridded_orbit.drop_vars(names=['lon', 'lat'])
        gridded_orbit = gridded_orbit.rename({'lon_bins':'lon', 'lat_bins':'lat', 'sample_time_bins':'time'})

    #If time not in bin_coordinates, run time independent binning / formatting
    else:
        # Project orbits data onto new coordinates using an nanmean operation in each bin
        binned_orbit_raw = flox.xarray.xarray_reduce(raw_orbit, masked_lon, masked_lat, func='nanmean', dim=('along_track', 'cross_track'),
                                                 fill_value=np.nan, expected_groups=(lon_coords_shifted, lat_coords_shifted), isbin=True)
    
        # Re-assign and rename dimension coordinates from pandas interval objects to the grid coordinate points
        # Remove lon and lat variables from dataset
        # Blank out high-error swath sections
        binned_orbit_1 = binned_orbit_raw.where((binned_orbit_raw.swath_blanking != np.nan) & (binned_orbit_raw.swath_blanking < 1))
        gridded_orbit = binned_orbit_1.assign_coords({'lon_bins':bin_coordinates.lon.values, 'lat_bins':bin_coordinates.lat.values})
        gridded_orbit = gridded_orbit.drop_vars(names=['lon', 'lat'])
        gridded_orbit = gridded_orbit.rename({'lon_bins':'lon', 'lat_bins':'lat'})
        
    return gridded_orbit

def genOrbits(num_orbits, start_time=datetime.datetime(2020, 1, 20, 0, 0, 0), end_time=datetime.datetime(2022, 1, 20, 0, 0, 0), region=None,
              uncertainty_path='../uncertainty_tables/odysea_sigma_vr_lut_height590km_look52deg_swath1672km.npz', bin_coordinates=None,
              concatanate=True, output_path=False, **kwargs):
    '''
    Wrapper function that automatically generates ODYSEA orbits and can output them or save them as files

    Args:
        num_orbits (int): Number of orbits to iterate through
        start_time (datetime.datetime): Start date for first orbit (passed to OdyseaSwath.getOrbits())
        end_time (datetime.datetime): End date for last orbit in generator (passed to OdyseaSwath.getOrbits())
        region (list): Optional region of interest in form of [lon_min, lon_max, lat_min, lat_max] (passed to (odysea_swath.OdyseaSwath())
        uncertainty_path (str): Path of uncertainty table (passed to errors.OdyseaErrors())
        bin_coordinates (xarray.dataset): Optionally regrids orbit datasets to specified lat-lon coordinates (passed to utils.gridOrbits())
        concatanate (boolean): If true, concatanate all orbits to a signal xarray dataset and return it
            - NOTE: VERY memory intensive
        output_path (str): If supplied, saves each orbit dataset as a netcdf file to specified folder
            - NOTE: To preserve multi-indices, datasets are encoded. They must be decoded using 
                   cfxr.decode_compress_to_multi_index(dataset, 'temporal')
        **kwargs: kwargs to be passed to colocate_model.GriddedModel()
    
    Returns:
        - If concatanate=True, single xarray dataset
        - If concatanate=False & output_path=False, list of xarray datasets
    '''
    
    # Initialize orbit generator object and orbit list
    odysea_swath = swath_sampling.OdyseaSwath(region=region)
    orbit_generator = odysea_swath.getOrbits(start_time, end_time)
    orbit_list = []
    # Check if var_selector was passed as kwarg -> If so, use it, if not, use default
    if 'var_selector' in kwargs:
        var_selector = kwargs['var_selector']
    else:
        var_selector = 'winds+currents'

    # Create GriddedModel and OdyseaErrors Objects
    model = colocate_model.GriddedModel(**kwargs)
    ody_errors = errors.OdyseaErrors(uncertainty_path)

    # Iterate through orbits
    for counter in range(0, num_orbits):

        orbit = next(orbit_generator)
        # Catch ValueError if orbit doesn't overlap region, skip itteration
        if isinstance(orbit, ValueError):
            continue     

        # Add data and errors
        if 'wind' in var_selector:
            orbit = model.colocateSwathWinds(orbit)
            orbit = ody_errors.setWindErrors(orbit)
        if 'current' in var_selector:
            orbit = model.colocateSwathCurrents(orbit)
            orbit = ody_errors.setCurrentErrors(orbit, etype='simulated_baseline')

        # Add orbit_index as dimension and coordinate to xarray dataset
        #orbit = orbit.assign_coords({'orbit_index':counter}).expand_dims('orbit_index')
        
        # Optionally regrid orbits individually
        if bin_coordinates is not None:
            orbit = gridOrbits(orbit, bin_coordinates, single_time=True)
            orbit_index_coords = np.ones_like(orbit.time.values, dtype='int')*counter
            orbit = orbit.assign_coords({'orbit_index':[counter]})
            orbit = orbit.stack({'temporal':('orbit_index','time')})
        
        # Optionally save orbits to file instead of returning them
        if output_path is not False:
            encoded_orbit = cfxr.encode_multi_index_as_compress(orbit, 'temporal')
            encoded_orbit.to_netcdf(os.path.join(output_path, 'orbit' + f'{counter:04d}' + '.nc'))

        
        # Append blanked orbit to orbit_list
        else:
            orbit_list.append(orbit)
    
    if concatanate is True:
        # Concatenate orbits in orbit_list into a single orbits dataset
        orbits = xr.concat(orbit_list, dim='temporal')
        return orbits
    elif output_path is False:
        # Return list of orbit datasets
        return orbit_list
    
def splineFactory(x,y,smoothing=.1):
    spl = UnivariateSpline(x, y)
    spl.set_smoothing_factor(.1)
    return spl

#@jit(nopython=True)
def signedAngleDiff(ang1,ang2):

    ang1 = np.asarray(ang1)
    ang2 = np.asarray(ang2)
    ang11 = normalizeTo360(ang1)
    ang22 = normalizeTo360(ang2)

    # ang11 = np.array(ang11)
    # ang21 = np.array(ang22)

    result = ang22 - ang11

    resultF = result.flatten()

    for ii in range(resultF.shape[0]):
        if resultF[ii] > 180:
            resultF[ii] = resultF[ii] - 360
        if resultF[ii] < -180:
            resultF[ii] = 360 + resultF[ii]

    result = resultF.reshape(np.shape(result))


    return result

def computeEncoderByXT(cross_track):
    """
    Compute the expected encoder angle from cross-track location
    Returns encoder_angle_fore, encoder_angle aft, the forward and backward looking samples. Degrees clockwise from the velocity vector.
    """
    
    encoder_angle_fore = normalizeTo180(90 - 180/np.pi*np.arccos(cross_track/np.max(cross_track)))

    encoder_angle_aft = normalizeTo180(180 - encoder_angle_fore)
 
    return encoder_angle_fore,encoder_angle_aft


def getBearing(platform_latitude,platform_longitude):

    d = 1
    X = np.zeros(np.shape(platform_latitude))
    Y = np.zeros(np.shape(platform_latitude))

    
    lon_diff = signedAngleDiff(platform_longitude[0:-d]*180/np.pi,platform_longitude[d::]*180/np.pi)*np.pi/180 # dont @ me
    
    
    X[d::] = np.cos(platform_latitude[d::]) * np.sin(lon_diff)
    Y[d::] = np.cos(platform_latitude[0:-d]) * np.sin(platform_latitude[d::]) - np.sin(platform_latitude[0:-d]) * np.cos(platform_latitude[d::]) * np.cos(lon_diff)

    pf_velocity_dir = np.arctan2(X,Y) * 180/np.pi

    return pf_velocity_dir


        
def localSTD(inpt,sigma):

    inpt[np.abs(inpt)>1] = np.nan
    inpt[np.isnan(inpt)] = 0

    u = scipy.ndimage.filters.gaussian_filter(np.copy(inpt), sigma=sigma)
    u2 = scipy.ndimage.filters.gaussian_filter(np.copy(inpt)**2, sigma=sigma)

    std = np.sqrt(u2 - u**2)

    return std
    

def normalizeTo180(angle):
    # note this strange logic was originally to use numba JIT
    for idx,ang in np.ndenumerate(angle):
    
        ang =  ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        angle[idx] = ang
        
    return angle


#@jit(nopython=True)
def normalizeTo180Jit(angle):

    for idx,ang in np.ndenumerate(angle):
    
        ang =  ang % 360
        ang = (ang + 360) % 360
        if ang > 180:
            ang -= 360
        angle[idx] = ang
        
    return angle


def normalizeTo360(angle):

    #angle2 = np.array(angle)

    angle2 = angle % 360

    return angle2


def fixLon(ds):

    ds['lon'].values = normalizeTo180(ds['lon'].values)

    return ds    


def toUTM(target_lon, target_lat):


    lonlat_epsg=4326  # This WGS 84 lat/lon
    xy_epsg=3857      # default google projection

    lonlat_epsg = lonlat_epsg
    lonlat_crs = CRS(lonlat_epsg)
    xy_epsg = xy_epsg
    xy_crs = CRS(xy_epsg)

    lonlat_to_xy = Transformer.from_crs(lonlat_crs,
                                         xy_crs,
                                         always_xy=True)

    target_x, target_y = lonlat_to_xy.transform(target_lon, target_lat)

    return target_x, target_y

def stressToWind(stress_magnitude):
    """
    Convert wind stress to wind field
    Ideally, this would be iterated to get Cd right
    """
    ### Assuming Large and Pond < 10 m/s
    cdl = 1.12e-3
    rho = 1.22  # density of the air

    wind_speed = np.sqrt(stress_magnitude/(rho*cdl))

    return wind_speed

 
def windToStress(wind_speed,wind_dir=None):
    """
    Convert wind to wind stress field
    Assumes current relative winds 
    """

    ### Assuming Large and Pond < 10 m/s
    cdl = 1.12e-3
    rho = 1.22  # density of the air
 
    #cd = (.49 + 0.065*wind_speed) * 10**-3
    #cd[wind_speed < 10] = cdl
    cd = cdl
    
    if wind_dir is None:
        stress_magnitude = rho * cd * wind_speed**2
        return stress_magnitude
    else:
        stress_magnitude = rho * cd * wind_speed**2
        stress_u = stress_magnitude * np.sin(wind_dir*np.pi/180)
        stress_v = stress_magnitude * np.cos(wind_dir*np.pi/180)
        return stress_u,stress_v

    
def SDToUVErrors(magnitude,direction,magnitude_error,direction_error):

    sin_dir = np.sin(direction*np.pi/180)
    cos_dir = np.cos(direction*np.pi/180)

    std_sin_dir = cos_dir*direction_error*np.pi/180
    std_cos_dir = sin_dir*direction_error*np.pi/180

    u_error = np.abs(sin_dir*magnitude)*np.sqrt((std_sin_dir/sin_dir)**2 + (magnitude_error/magnitude)**2)
    v_error = np.abs(cos_dir*magnitude)*np.sqrt((std_cos_dir/cos_dir)**2 + (magnitude_error/magnitude)**2)

    return u_error,v_error



def makePlot(lon,lat,data,vmin,vmax,cblabel,colormap,figsize=(20,10),bg=True,gridMe=False,is_err=False,globe=False,cb=True):

    
    if gridMe:
        
        mask = np.isfinite(lon+lat+data)

        
        lon_lin = np.arange(-180,180,0.25)
        lat_lin = np.arange(-90,90,0.25)

        lon_mesh,lat_mesh = np.meshgrid((lon_lin[1::]+lon_lin[0:-1])/2,(lat_lin[1::]+lat_lin[0:-1])/2)
        
        data, bin_edges, binnumber = scipy.stats.binned_statistic_dd([lon[mask],lat[mask]],values=data[mask],statistic='mean',bins=[lon_lin,lat_lin])
        
            
        data = data.T
        lon = lon_mesh
        lat = lat_mesh
    
    fig = plt.figure(figsize=figsize)
    if globe:
        ax = plt.subplot(111, projection=ccrs.Orthographic(-65, 15))
    else:
        ax = plt.subplot(111, projection=ccrs.PlateCarree())

    if bg:
        ax.background_img(name='BM', resolution='low')

    
    plt.pcolormesh(lon, lat, data,transform=ccrs.PlateCarree(),vmax=vmax,vmin=vmin,cmap=colormap)

    ax.coastlines()
    
    if cb:
        plt.colorbar(label=cblabel,orientation='horizontal',fraction=0.046, pad=0.04)

    fig.tight_layout()

    
    plt.show()