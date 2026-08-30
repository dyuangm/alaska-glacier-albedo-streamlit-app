import base64
import io
import os
import tempfile
import datetime

import folium
import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st
import xarray as xr
import fsspec
from folium.raster_layers import ImageOverlay
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPolygon
from streamlit_folium import st_folium

GLACIER_INDEX_PATH = "glacier_index_v2.parquet"
DEFAULT_CENTER = [64.2, -149.5]
DEFAULT_ZOOM = 5
GLACIER_ZOOM = 11
ALBEDO_VAR = "albedo"


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
def _cube_date_range(ds: xr.Dataset) -> tuple[datetime.date | None, datetime.date | None]:
    """Return the (min, max) date covered by a cube's time/year coordinate."""
    if "time" in ds.coords:
        days = np.sort(ds["time"].values.astype("datetime64[D]"))
        if len(days):
            return days[0].astype(object), days[-1].astype(object)
    elif "year" in ds.coords:
        years = np.sort(ds["year"].values.astype(int))
        if len(years):
            return (
                datetime.date(int(years[0]), 1, 1),
                datetime.date(int(years[-1]), 12, 31),
            )
    return None, None


@st.cache_data(show_spinner=False)
def prepare_local_cube(ref_json_uri: str, glacier_name: str) -> dict:
    """Download a glacier's full albedo cube to local disk once and compute map bounds.

    The source is a Kerchunk reference JSON in GCS that exposes a remote NetCDF as a
    Zarr store. The whole 3D (time, y, x) cube is cached on disk so that slider moves
    only need a cheap local slice. Returns a dict with ``local_path``, ``bounds`` and
    the axis-orientation flags, or ``{"error": ...}`` on failure.
    """
    gcp_creds = dict(st.secrets["gcp_service_account"])
    safe_name = glacier_name.replace(" ", "_").replace("/", "")
    local_path = os.path.join(tempfile.gettempdir(), f"{safe_name}_albedo.nc")

    try:
        if not os.path.exists(local_path):
            mapper = fsspec.get_mapper(
                "reference://",
                fo=ref_json_uri,
                target_protocol="gcs",
                asynchronous=False,
                target_options={"token": gcp_creds},
                remote_options={"token": gcp_creds},
            )
            with xr.open_dataset(
                mapper, engine="zarr", backend_kwargs={"consolidated": False}
            ) as ds:
                for var in ds.variables:
                    ds[var].encoding.clear()
                ds.to_netcdf(local_path, engine="h5netcdf")

        with xr.open_dataset(local_path) as ds:
            x_name = "x" if "x" in ds.coords else "easting"
            y_name = "y" if "y" in ds.coords else "northing"
            x, y = ds.coords[x_name].values, ds.coords[y_name].values

            src_crs = next(
                (
                    ds[var].attrs.get("wkt") or ds[var].attrs.get("spatial_ref")
                    for var in ds.data_vars
                    if "wkt" in ds[var].attrs or "spatial_ref" in ds[var].attrs
                ),
                "EPSG:32607",
            )

            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            lon_nw, lat_nw = transformer.transform(x[0], y[-1])
            lon_se, lat_se = transformer.transform(x[-1], y[0])
            bounds = [[float(lat_se), float(lon_nw)], [float(lat_nw), float(lon_se)]]

            date_min, date_max = _cube_date_range(ds)

        return {
            "local_path": local_path,
            "bounds": bounds,
            "flip_y": bool(y[0] < y[-1]),
            "flip_x": bool(x[0] > x[-1]),
            "date_min": date_min,
            "date_max": date_max,
        }

    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        return {"error": str(exc)}


def get_local_slice(cube_meta: dict, date_str: str, var_name: str = ALBEDO_VAR) -> np.ndarray:
    """Pull a single 2D time slice from the locally cached NetCDF cube."""
    with xr.open_dataset(cube_meta["local_path"]) as ds:
        if "time" in ds.coords:
            time_slice = ds[var_name].sel(time=date_str, method="nearest")
        elif "year" in ds.coords:
            time_slice = ds[var_name].sel(year=int(date_str[:4]))
        else:
            time_slice = ds[var_name].isel(time=0)
        data = time_slice.values

    if cube_meta["flip_y"]:
        data = np.flipud(data)
    if cube_meta["flip_x"]:
        data = np.fliplr(data)
    return data


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def array_to_png_data_url(
    data: np.ndarray, vmin: float = 0.1, vmax: float = 0.9, cmap_name: str = "Greys_r"
) -> str | None:
    """Colorize a 2D array and return it as a base64 PNG data URL for a map overlay."""
    data = np.squeeze(data)
    if data.ndim == 1:
        data = data.reshape(data.shape[0], 1)
    elif data.ndim != 2:
        return None

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    try:
        cmap = mpl.colormaps[cmap_name]
    except KeyError:
        cmap = mpl.colormaps["viridis"]

    rgba = cmap(norm(data))
    mask = np.isnan(data) | (data < vmin) | (data > vmax)
    rgba[mask, 3] = 0.0

    img = Image.fromarray((rgba * 255).astype(np.uint8), mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"


@st.cache_data
def load_glacier_data() -> gpd.GeoDataFrame:
    """Load the glacier index, normalize geometries, and reproject to EPSG:4326."""

    def clean_geometry(geom):
        if geom is None or geom.geom_type in ("Polygon", "MultiPolygon"):
            return geom
        if geom.geom_type == "GeometryCollection":
            polys = []
            for g in geom.geoms:
                if g.geom_type == "Polygon":
                    polys.append(g)
                elif g.geom_type == "MultiPolygon":
                    polys.extend(g.geoms)
            if polys:
                return MultiPolygon(polys)
        return geom

    gdf = gpd.read_parquet(GLACIER_INDEX_PATH)
    gdf["geometry"] = gdf["geometry"].apply(clean_geometry)
    keep_cols = ["glacier_name", "gcs_uri", "geometry", "rgi_id"]
    gdf = gdf[[c for c in keep_cols if c in gdf.columns]]
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Alaska Glacier Albedo Explorer", layout="wide")

gdf = load_glacier_data()
glacier_names = sorted(gdf["glacier_name"].dropna().unique().tolist())

st.session_state.setdefault("map_center", DEFAULT_CENTER)
st.session_state.setdefault("map_zoom", DEFAULT_ZOOM)
st.session_state.setdefault("move_camera", False)
st.session_state.setdefault("selected_date", datetime.date(2023, 7, 15))


def fly_to_glacier() -> None:
    """Recenter the map on the selected glacier (or back to Alaska if cleared)."""
    selected = st.session_state.glacier_search
    if selected:
        centroid = gdf.loc[gdf["glacier_name"] == selected].iloc[0].geometry.centroid
        st.session_state.map_center = [centroid.y, centroid.x]
        st.session_state.map_zoom = GLACIER_ZOOM
    else:
        st.session_state.map_center = DEFAULT_CENTER
        st.session_state.map_zoom = DEFAULT_ZOOM
    st.session_state.move_camera = True


def request_camera_move() -> None:
    st.session_state.move_camera = True


st.title("Alaska Glacier Albedo Explorer")
st.markdown(
    "Explore daily albedo data for glaciers across Alaska. Use the sidebar to search "
    "for a specific glacier and the slider below to move through time."
)

with st.sidebar:
    st.header("Controls")

    selected_glacier = st.selectbox(
        "Search for a Glacier:",
        options=glacier_names,
        key="glacier_search",
        help="Type or select a glacier to zoom in.",
        on_change=fly_to_glacier,
    )

# Prepare the glacier's data cube once; prepare_local_cube is cached so this is
# cheap on reruns and shared by the date slider, the download button and the map.
cube_meta = None
if selected_glacier:
    gcs_uri = gdf.loc[gdf["glacier_name"] == selected_glacier].iloc[0]["gcs_uri"]
    with st.spinner(f"Preparing data cube for {selected_glacier}..."):
        cube_meta = prepare_local_cube(gcs_uri, selected_glacier)

# Date selector: full width above the map, with its range derived from the
# selected glacier's cube. Disabled until a glacier (with valid dates) is loaded.
cube_dates_ok = bool(
    cube_meta
    and "error" not in cube_meta
    and cube_meta["date_min"]
    and cube_meta["date_max"]
    and cube_meta["date_min"] < cube_meta["date_max"]
)

if cube_dates_ok:
    date_min, date_max = cube_meta["date_min"], cube_meta["date_max"]
    current = min(max(st.session_state.selected_date, date_min), date_max)
    selected_date = st.slider(
        "Select Date (Daily Resolution):",
        min_value=date_min,
        max_value=date_max,
        value=current,
        format="YYYY-MM-DD",
    )
    if selected_date != st.session_state.selected_date:
        st.session_state.selected_date = selected_date
        request_camera_move()
else:
    ph_min, ph_max = datetime.date(2019, 1, 1), datetime.date(2025, 12, 31)
    st.slider(
        "Select Date (Daily Resolution):",
        min_value=ph_min,
        max_value=ph_max,
        value=min(max(st.session_state.selected_date, ph_min), ph_max),
        format="YYYY-MM-DD",
        disabled=True,
        help="Select a glacier to enable the date selector.",
    )
    selected_date = None

with st.sidebar:
    st.divider()
    st.subheader("Export")
    if not selected_glacier:
        st.markdown("*Select a glacier to enable downloads.*")
    elif cube_meta and "error" not in cube_meta:
        with open(cube_meta["local_path"], "rb") as f:
            st.download_button(
                label=f"Download {selected_glacier} NetCDF",
                data=f,
                file_name=f"{selected_glacier.replace(' ', '_')}_albedo.nc",
                mime="application/x-netcdf",
            )
    else:
        st.error(f"Error fetching data: {cube_meta['error']}")

# ---------------------------------------------------------------------------
# Map
# ---------------------------------------------------------------------------
m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles="OpenStreetMap")

if selected_date and cube_meta and "error" not in cube_meta:
    albedo_data = get_local_slice(cube_meta, selected_date.strftime("%Y-%m-%d"))
    png_url = array_to_png_data_url(albedo_data)
    if png_url:
        ImageOverlay(
            image=png_url,
            bounds=cube_meta["bounds"],
            opacity=0.85,
            interactive=False,
            cross_origin=False,
            z_index=1,
        ).add_to(m)
elif cube_meta and "error" in cube_meta:
    st.error(f"Failed to load data: {cube_meta['error']}")

if selected_glacier:
    glacier_gdf = gdf[gdf["glacier_name"] == selected_glacier]
    folium.GeoJson(
        glacier_gdf.to_json(),
        name="Glacier Outline",
        style_function=lambda _: {"color": "#1f77b4", "weight": 2, "fillOpacity": 0.1},
        tooltip=folium.GeoJsonTooltip(
            fields=["glacier_name", "rgi_id"],
            aliases=["Glacier Name:", "RGI ID:"],
            style=(
                "background-color: white; color: #333333; font-family: arial; "
                "font-size: 12px; padding: 10px;"
            ),
        ),
    ).add_to(m)

# Only push center/zoom to the widget on an explicit fly-to or slider move so that
# manual panning stays responsive.
center_arg = st.session_state.map_center if st.session_state.move_camera else None
zoom_arg = st.session_state.map_zoom if st.session_state.move_camera else None

st_data = st_folium(
    m,
    use_container_width=True,
    height=600,
    center=center_arg,
    zoom=zoom_arg,
    returned_objects=["center", "zoom"],
)

# Track the user's manual panning between reruns.
if st_data and st_data.get("center"):
    st.session_state.map_center = [st_data["center"]["lat"], st_data["center"]["lng"]]
    st.session_state.map_zoom = st_data["zoom"]

st.session_state.move_camera = False
