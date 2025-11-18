import os
import xarray as xr

ANDES_BOUNDS = [-32, -14, 360-78, 360-62]
HIMALAYAS_BOUNDS = [15, 55, 60, 120]

data_dir = '~/lehre/msc-intro-computational-meteorology-exercises-w2025/ERA5'
t2m_wind_10m_path = os.path.join(data_dir, 'ERA5_monthly.1940-2025.t2m_wind10_pres.1deg.nc')
precip_radflx_path = os.path.join(data_dir, 'ERA5_monthly.1940-2025.precip_radflx.1deg.nc')
daily_1deg_88_94_path = os.path.join(data_dir, 'ERA5_daily.1988-1994.1deg.nc')
daily_1deg_79_85_path = os.path.join(data_dir, 'ERA5_daily.1979-1985.1deg.nc')
daily_1deg_60_66_path = os.path.join(data_dir, 'ERA5_daily.1960-1966.1deg.nc')
elevation_data_path = 'elevation_data/elevation_data_low_res.nc'

def open_dataset(data_path):
    return xr.open_dataset(data_path)

def load_elevation_data():
    return xr.open_dataset(elevation_data_path)['z']

def get_elevation_data(elevation_data, bounds=ANDES_BOUNDS):
    lat_min, lat_max, lon_min, lon_max = bounds
    
    if (elevation_data.lon > 180).any():
        elevation_data = elevation_data.assign_coords(
            lon=((elevation_data.lon + 180) % 360) - 180
        ).sortby('lon')
    
    lon_min_180 = ((lon_min + 180) % 360) - 180
    lon_max_180 = ((lon_max + 180) % 360) - 180
    
    lat_coords = elevation_data.lat.values
    lat_ascending = lat_coords[0] < lat_coords[-1]
    
    lon_span = lon_max - lon_min
    if lon_span >= 359:
        elevation = elevation_data.sel(
            lat=slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
        )
    elif lon_min_180 > lon_max_180:
        # Wrapping around dateline
        elevation = xr.concat([
            elevation_data.sel(
                lon=slice(lon_min_180, 180),
                lat=slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
            ),
            elevation_data.sel(
                lon=slice(-180, lon_max_180),
                lat=slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
            )
        ], dim='lon')
    else:
        elevation = elevation_data.sel(
            lon=slice(lon_min_180, lon_max_180),
            lat=slice(lat_min, lat_max) if lat_ascending else slice(lat_max, lat_min)
        )
    if elevation.size == 0:
        print(f"Warning: No elevation data found for bounds {bounds}")
        return None
    
    return elevation

def mask_and_filter(ds, by_time=False, time=None, by_latlon=False, bounds=ANDES_BOUNDS):
    if by_time and time is None:
        raise ValueError("If filtering by time, a timestep must be supplied (time=?)")
    lat_min, lat_max, lon_min, lon_max = bounds

    ds_out = ds

    if by_time:
        try:
            ds_out = ds.isel(valid_time=time)
        except Exception:
            ds_out = ds.isel(time=time)

    if by_latlon:
        where_clause = (ds_out.lat >= lat_min) & (ds_out.lat <= lat_max) & (ds_out.lon >= lon_min) & (ds_out.lon <= lon_max)
        ds_out = ds_out.where(where_clause, drop=True)
    
    return ds_out