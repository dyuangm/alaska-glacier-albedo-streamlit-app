import io
import base64
import streamlit as st
from shapely.geometry import Polygon, MultiPolygon
import geopandas as gpd
import pandas as pd
import numpy as np
import folium
import os
from folium.raster_layers import ImageOverlay
from streamlit_folium import st_folium
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
import fsspec
import datetime
import xarray as xr
from pyproj import Transformer

# @st.cache_data(show_spinner=False)
# def fetch_kerchunk_slice(ref_json_uri: str, str_date: str, var_name: str = "albedo"):
#     """Fetches a single glacier's time slice using Kerchunk."""
#     try:
#         mapper = fsspec.get_mapper(
#             "reference://",
#             fo=ref_json_uri,
#             target_protocol="gcs",
#             target_options={"token": None}, # Auth to read the .json file
#             remote_options={"token": None}  # Auth to read the .nc file bytes
#         )        
#         with xr.open_dataset(mapper, engine="zarr", backend_kwargs={"consolidated": False}) as ds:
#             if "time" in ds.coords: time_slice = ds[var_name].sel(time=str_date, method="nearest")
#             else: time_slice = ds[var_name].isel(time=0)

#             x_name = "x" if "x" in ds.coords else "easting"
#             y_name = "y" if "y" in ds.coords else "northing"
#             x, y = ds.coords[x_name].values, ds.coords[y_name].values
#             albedo_data = time_slice.values

#             src_crs_wkt = None
#             for var in ds.data_vars:
#                 if 'wkt' in ds[var].attrs: src_crs_wkt = ds[var].attrs['wkt']; break
#                 if 'spatial_ref' in ds[var].attrs: src_crs_wkt = ds[var].attrs['spatial_ref']; break
#             src_crs = src_crs_wkt if src_crs_wkt else "EPSG:32607"

#             if y[0] < y[-1]: y = y[::-1]; albedo_data = np.flipud(albedo_data)
#             if x[0] > x[-1]: x = x[::-1]; albedo_data = np.fliplr(albedo_data)

#             min_x, max_x = x[0], x[-1]
#             min_y, max_y = y[0], y[-1]

#             transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
#             lon_nw, lat_nw = transformer.transform(min_x, max_y)
#             lon_se, lat_se = transformer.transform(max_x, min_y)

#             bounds = [[float(lat_se), float(lon_nw)], [float(lat_nw), float(lon_se)]]
#             return {"uri": ref_json_uri, "bounds": bounds, "data": albedo_data}
            
#     except Exception as e:
#         return {"uri": ref_json_uri, "error": str(e)}

@st.cache_data(show_spinner=False)
def prepare_local_cube(ref_json_uri: str, glacier_name: str):
    """Downloads the full cube to disk once and computes map bounds."""
    safe_name = glacier_name.replace(" ", "_").replace("/", "")
    local_path = f"{safe_name}_albedo.nc"
    
    try:
        # 1. Download to local disk if it doesn't exist yet
        if not os.path.exists(local_path):
            mapper = fsspec.get_mapper(
                "reference://", fo=ref_json_uri, target_protocol="gcs",
                asynchronous=False, target_options={"asynchronous": False},
                remote_options={"asynchronous": False}
            )            
            with xr.open_dataset(mapper, engine="zarr", backend_kwargs={"consolidated": False}) as ds:
                for var in ds.variables:
                    ds[var].encoding.clear()
                ds.to_netcdf(local_path, engine="h5netcdf") # Saves the entire 3D cube locally
                
        # 2. Compute bounds once so we don't recalculate on every slider move
        with xr.open_dataset(local_path) as ds:
            x_name = "x" if "x" in ds.coords else "easting"
            y_name = "y" if "y" in ds.coords else "northing"
            x, y = ds.coords[x_name].values, ds.coords[y_name].values
            
            src_crs_wkt = next((ds[var].attrs.get('wkt') or ds[var].attrs.get('spatial_ref') 
                                for var in ds.data_vars if 'wkt' in ds[var].attrs or 'spatial_ref' in ds[var].attrs), "EPSG:32607")
            
            transformer = Transformer.from_crs(src_crs_wkt, "EPSG:4326", always_xy=True)
            lon_nw, lat_nw = transformer.transform(x[0], y[-1])
            lon_se, lat_se = transformer.transform(x[-1], y[0])
            
            bounds = [[float(lat_se), float(lon_nw)], [float(lat_nw), float(lon_se)]]
            
        return {
            "local_path": local_path, 
            "bounds": bounds, 
            "flip_y": bool(y[0] < y[-1]), 
            "flip_x": bool(x[0] > x[-1])
        }
        
    except Exception as e:
        return {"error": str(e)}

def get_local_slice(cube_meta: dict, date_str: str, var_name: str = "albedo"):
    """Instantly slices the local NetCDF file."""
    with xr.open_dataset(cube_meta["local_path"]) as ds:
        if "time" in ds.coords: time_slice = ds[var_name].sel(time=date_str, method="nearest")
        elif "year" in ds.coords: time_slice = ds[var_name].sel(year=int(date_str[:4]))
        else: time_slice = ds[var_name].isel(time=0)
        
        albedo_data = time_slice.values
        
    if cube_meta["flip_y"]: albedo_data = np.flipud(albedo_data)
    if cube_meta["flip_x"]: albedo_data = np.fliplr(albedo_data)
    
    return albedo_data

def array_to_png_data_url(data: np.ndarray, vmin=0.1, vmax=0.9, cmap_name="Greys_r"):
    data = np.squeeze(data)
    if data.ndim == 1: data = data.reshape(data.shape[0], 1)
    elif data.ndim != 2: return None

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    try: cmap = cm.get_cmap(cmap_name)
    except ValueError: cmap = cm.get_cmap("viridis")
    
    rgba = cmap(norm(data))
    mask = np.isnan(data) | (data < vmin) | (data > vmax)
    rgba[mask, 3] = 0.0  
    
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    img = Image.fromarray(rgba_uint8, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


st.set_page_config(
    page_title="Alaska Glacier Albedo Explorer",
    layout="wide"
)

@st.cache_data
def load_glacier_data():
    gdf = gpd.read_parquet("glacier_index_v2.parquet")
    def clean_geometry(geom):
        if geom is None:
            return None
        if geom.geom_type in ['Polygon', 'MultiPolygon']:
            return geom
        if geom.geom_type == 'GeometryCollection':
            polys = []
            for g in geom.geoms:
                if g.geom_type == 'Polygon':
                    polys.append(g)
                elif g.geom_type == 'MultiPolygon':
                    polys.extend(list(g.geoms))
            if polys:
                return MultiPolygon(polys)
        return geom           
    gdf['geometry'] = gdf['geometry'].apply(clean_geometry)
    keep_cols = ['glacier_name', 'gcs_uri', 'geometry', 'rgi_id']
    gdf = gdf[[c for c in keep_cols if c in gdf.columns]]
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf

gdf = load_glacier_data()

glacier_names = sorted(gdf['glacier_name'].dropna().unique().tolist())

if "map_center" not in st.session_state:
    st.session_state.map_center = [64.2, -149.5]
if "map_zoom" not in st.session_state:
    st.session_state.map_zoom = 5
if "move_camera" not in st.session_state:
    st.session_state.move_camera = False  # NEW FLAG
    
def fly_to_glacier():
    selected = st.session_state.glacier_search
    if selected:
        glacier_row = gdf[gdf['glacier_name'] == selected].iloc[0]
        centroid = glacier_row.geometry.centroid
        st.session_state.map_center = [centroid.y, centroid.x]
        st.session_state.map_zoom = 11
        st.session_state.move_camera = True # Trigger the map to move
    else:
        st.session_state.map_center = [64.2, -149.5]
        st.session_state.map_zoom = 5
        st.session_state.move_camera = True

st.title("Alaska Glacier Albedo Explorer")
st.markdown(
    "Explore daily albedo data for glaciers across Alaska. Use the sidebar to search "
    "for a specific glacier and across time."
)

with st.sidebar:
    st.header("Controls")
    
    selected_glacier = st.selectbox(
        "Search for a Glacier:",
        options=glacier_names,
        key="glacier_search",
        help="Type or select a glacier to zoom in.",
        on_change=fly_to_glacier
    )
    
    selected_date = st.slider(
        "Select Date (Daily Resolution):",
        min_value=datetime.date(2019, 1, 1),
        max_value=datetime.date(2025, 12, 31),
        value=datetime.date(2023, 7, 15),
        format="YYYY-MM-DD"
    )
    
    st.divider()
    
    st.subheader("Export")
    st.markdown("*Select a glacier to enable downloads.*")
    if selected_glacier:
        gcs_uri = gdf[gdf['glacier_name'] == selected_glacier].iloc[0]['gcs_uri']
        with st.spinner(f"Downloading full cube for {selected_glacier}..."):
            cube_meta = prepare_local_cube(gcs_uri, selected_glacier)
        if "error" not in cube_meta:
            with open(cube_meta["local_path"], "rb") as f:
                st.download_button(
                    label=f"Download {selected_glacier} NetCDF",
                    data=f,
                    file_name=f"{selected_glacier.replace(' ', '_')}_albedo.nc",
                    mime="application/x-netcdf"
                )
        else:
            st.error(f"Error fetching data: {cube_meta['error']}")
    else:
        st.markdown("*Select a glacier to enable downloads.*")
        
# m = folium.Map(location=[64.2, -149.5], zoom_start=5, tiles="CartoDB Voyager")
m = folium.Map(
    location=st.session_state.map_center, 
    zoom_start=st.session_state.map_zoom, 
    tiles="CartoDB Voyager"
)
# Assuming `selected_glacier` and `selected_date` are defined from the sidebar
active_raster = None
if st.session_state.glacier_search:
    selected_uri = gdf[gdf["glacier_name"] == st.session_state.glacier_search].iloc[0]["gcs_uri"]
    with st.spinner(f"Preparing full data cube for {st.session_state.glacier_search}..."):
        cube_meta = prepare_local_cube(selected_uri, st.session_state.glacier_search)
        
    if "error" not in cube_meta:
        # 2. Format the slider's date object into a string pandas/xarray understands
        formatted_date = selected_date.strftime("%Y-%m-%d")
        
        # 3. Instantly pull the 2D slice from the local file
        albedo_data = get_local_slice(cube_meta, formatted_date)
        
        # 4. Colorize and convert to PNG
        png_url = array_to_png_data_url(albedo_data)
        if png_url:
            active_raster = {"url": png_url, "bounds": cube_meta["bounds"]}
    else:
        st.error(f"Failed to load data: {cube_meta['error']}")    # with st.spinner(f"Fetching albedo cube for {st.session_state.glacier_search}..."):
    #     cube_result = fetch_kerchunk_slice(selected_uri, selected_date.strftime("%Y-%m-%d"))
    #     print(cube_result)
    #     if "error" not in cube_result:
    #         png_url = array_to_png_data_url(cube_result["data"])
    #         if png_url:
    #             active_raster = {"url": png_url, "bounds": cube_result["bounds"]}
    #     else:
    #         st.error(f"Failed to load data: {cube_result['error']}")
    
if active_raster:
    ImageOverlay(
        image=active_raster["url"],
        bounds=active_raster["bounds"],
        opacity=0.85,
        interactive=False,
        cross_origin=False,
        z_index=1
    ).add_to(m)

if st.session_state.glacier_search:
    glacier_gdf = gdf[gdf['glacier_name'] == st.session_state.glacier_search]
    tooltip = folium.GeoJsonTooltip(
        fields=['glacier_name', 'rgi_id'],
        aliases=['Glacier Name:', 'RGI ID:'],
        style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;")
    )
    folium.GeoJson(
        glacier_gdf.to_json(),
        name="Glacier Outline",
        style_function=lambda x: {'color': '#1f77b4', 'weight': 2, 'fillOpacity': 0.1},
        tooltip=tooltip
    ).add_to(m)

center_arg = st.session_state.map_center if st.session_state.move_camera else None
zoom_arg = st.session_state.map_zoom if st.session_state.move_camera else None

st_data = st_folium(
    m, 
    use_container_width=True, 
    height=600,
    center=center_arg, 
    zoom=zoom_arg,     
    returned_objects=["center", "zoom"] # MUST be captured to preserve manual panning
)

if st_data and st_data.get("center"):
    st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    st.session_state.map_zoom = st_data["zoom"]
if st.session_state.move_camera:
    st.session_state.move_camera = False