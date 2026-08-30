import base64
import datetime
import glob
import io
import json
import os
import tempfile
import threading

import geopandas as gpd
import matplotlib as mpl
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import xarray as xr
import fsspec
from google.cloud import storage
from PIL import Image
from pyproj import Transformer
from shapely.geometry import MultiPolygon

GLACIER_INDEX_PATH = "glacier_index_v2.parquet"
CUBE_INDEX_PATH = "glacier_index.parquet"  # v1 index: gcs_uri points at the raw .nc
JSON_DIR = "s2_json"
CUBE_DIR = "s2_glacier_cubes_slope_dem"
SIGNED_URL_TTL = datetime.timedelta(minutes=15)
ALASKA_VIEW = {"center": [64.2, -149.5], "zoom": 5}
ALBEDO_VAR = "albedo"
ALBEDO_VMIN = 0.0
ALBEDO_VMAX = 1.0
ALBEDO_CMAP = "Greys_r"
OVERLAY_OPACITY = 0.85
MAP_HEIGHT = 600
CONTROLS_HEIGHT = 56

CUBE_SUFFIX = "_albedo.nc"
MAX_LOCAL_CUBES = 3  # most-recently-used cube files kept on disk
FRAMES_CACHE_ENTRIES = 6  # glaciers' worth of rendered frames kept in memory
FRAMES_CACHE_TTL = 3600


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _storage_client() -> storage.Client:
    return storage.Client.from_service_account_info(
        dict(st.secrets["gcp_service_account"])
    )


@st.cache_data(ttl=600, show_spinner=False)
def signed_download_url(cube_uri: str) -> str:
    """A short-lived v4 signed URL so the browser pulls the raw cube straight from GCS.

    ``cube_uri`` is a ``gs://bucket/path/glacier.nc`` URI. Cached for less than the
    link's own lifetime so a stale URL is never handed out.
    """
    bucket_name, _, blob_name = cube_uri.removeprefix("gs://").partition("/")
    blob = _storage_client().bucket(bucket_name).blob(blob_name)
    return blob.generate_signed_url(
        version="v4", expiration=SIGNED_URL_TTL, method="GET"
    )


_download_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: str) -> threading.Lock:
    """Per-path lock so concurrent sessions don't download the same cube twice."""
    with _locks_guard:
        return _download_locks.setdefault(path, threading.Lock())


def _prune_cube_cache(keep: int = MAX_LOCAL_CUBES) -> None:
    """Delete all but the ``keep`` most-recently-used cube files in the temp dir."""
    files = glob.glob(os.path.join(tempfile.gettempdir(), f"*{CUBE_SUFFIX}"))
    for path in sorted(files, key=os.path.getmtime, reverse=True)[keep:]:
        try:
            os.remove(path)
        except OSError:
            pass


def prepare_local_cube(ref_json_uri: str, glacier_id: str) -> dict:
    """Ensure a glacier's full albedo cube is on local disk and compute map bounds.

    ``glacier_id`` is any stable, filesystem-safe key for the glacier (its RGI ID).
    The source is a Kerchunk reference JSON in GCS that exposes a remote NetCDF as a
    Zarr store. Cube files are kept in an LRU of at most ``MAX_LOCAL_CUBES`` on disk;
    this is not Streamlit-cached (``build_frames`` is, and holds the only long-lived
    result). Returns ``local_path``/``bounds``/flip flags, or ``{"error": ...}``.
    """
    gcp_creds = dict(st.secrets["gcp_service_account"])
    safe_name = glacier_id.replace(" ", "_").replace("/", "")
    local_path = os.path.join(tempfile.gettempdir(), f"{safe_name}{CUBE_SUFFIX}")

    try:
        with _lock_for(local_path):
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
            else:
                os.utime(local_path, None)  # mark as most-recently-used
            _prune_cube_cache()

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

        return {
            "local_path": local_path,
            "bounds": bounds,
            "flip_y": bool(y[0] < y[-1]),
            "flip_x": bool(x[0] > x[-1]),
        }

    except Exception as exc:  # noqa: BLE001 - surfaced to the user in the UI
        return {"error": str(exc)}


def _time_labels(ds: xr.Dataset) -> list[str]:
    """Human-readable label for every time step in the cube."""
    if "time" in ds.coords:
        days = ds["time"].values.astype("datetime64[D]")
        return [str(d) for d in days]
    if "year" in ds.coords:
        return [str(int(v)) for v in ds["year"].values]
    return ["0"]


@st.cache_data(
    show_spinner="Rendering albedo frames...",
    max_entries=FRAMES_CACHE_ENTRIES,
    ttl=FRAMES_CACHE_TTL,
)
def build_frames(ref_json_uri: str, glacier_id: str) -> dict:
    """Colorize every time step of a glacier's cube into a PNG data URL.

    Returns ``{"bounds": [...], "frames": [{"label": str, "url": str}, ...]}`` so the
    browser can scrub through time without any Streamlit reruns, or ``{"error": ...}``.
    """
    cube_meta = prepare_local_cube(ref_json_uri, glacier_id)
    if "error" in cube_meta:
        return cube_meta

    try:
        with xr.open_dataset(cube_meta["local_path"]) as ds:
            labels = _time_labels(ds)
            cube = np.asarray(ds[ALBEDO_VAR].values, dtype="float32")

        if cube.ndim == 2:  # single time step
            cube = cube[None, ...]
        if cube_meta["flip_y"]:
            cube = cube[:, ::-1, :]
        if cube_meta["flip_x"]:
            cube = cube[:, :, ::-1]

        frames = [
            {"label": label, "url": array_to_png_data_url(cube[i])}
            for i, label in enumerate(labels[: cube.shape[0]])
        ]
        frames = [f for f in frames if f["url"]]
        return {"bounds": cube_meta["bounds"], "frames": frames}

    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def _colormap_css_stops(cmap_name: str = ALBEDO_CMAP, n: int = 24) -> list[str]:
    """Sample a matplotlib colormap into hex stops spanning vmin -> vmax."""
    try:
        cmap = mpl.colormaps[cmap_name]
    except KeyError:
        cmap = mpl.colormaps["viridis"]
    return [mcolors.to_hex(cmap(i / (n - 1))) for i in range(n)]


def array_to_png_data_url(
    data: np.ndarray,
    vmin: float = ALBEDO_VMIN,
    vmax: float = ALBEDO_VMAX,
    cmap_name: str = ALBEDO_CMAP,
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


def _resolve_cube_uris(gdf: gpd.GeoDataFrame) -> pd.Series:
    """Raw NetCDF cube URI per glacier.

    The v2 index stores the Kerchunk reference JSON in ``gcs_uri``; the raw cube sits
    in a sibling directory. Prefer the authoritative URI from the v1 index (joined on
    ``rgi_id``), falling back to rewriting the JSON path.
    """
    derived = gdf["gcs_uri"].str.replace(
        f"/{JSON_DIR}/", f"/{CUBE_DIR}/", regex=False
    ).str.replace(r"\.json$", ".nc", regex=True)

    if os.path.exists(CUBE_INDEX_PATH) and "rgi_id" in gdf.columns:
        v1 = pd.read_parquet(CUBE_INDEX_PATH, columns=["rgi_id", "gcs_uri"])
        lookup = dict(zip(v1["rgi_id"], v1["gcs_uri"]))
        return gdf["rgi_id"].map(lookup).fillna(derived)
    return derived


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
    gdf["cube_uri"] = _resolve_cube_uris(gdf)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf


def render_map(frame_data: dict | None, outline_geojson: dict | None) -> None:
    """Render a self-contained Leaflet map with a client-side time slider.

    The map only reloads when its HTML changes (i.e. a different glacier). Moving the
    slider swaps the single image overlay in the browser with no Streamlit rerun.
    """
    frames = (frame_data or {}).get("frames", [])
    bounds = (frame_data or {}).get("bounds")

    gradient = "linear-gradient(to top, " + ", ".join(_colormap_css_stops()) + ")"
    payload = json.dumps(
        {
            "frames": frames,
            "bounds": bounds,
            "outline": outline_geojson,
            "opacity": OVERLAY_OPACITY,
            "alaska": ALASKA_VIEW,
            "mapHeight": MAP_HEIGHT,
            "legend": {
                "title": "Albedo",
                "gradient": gradient,
                "ticks": [
                    f"{v:.2f}" for v in np.linspace(ALBEDO_VMAX, ALBEDO_VMIN, 5)
                ],
            },
        }
    )

    html = _MAP_TEMPLATE.replace("__PAYLOAD__", payload)
    components.html(html, height=MAP_HEIGHT + CONTROLS_HEIGHT)


_MAP_TEMPLATE = """
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body { margin: 0; font-family: -apple-system, system-ui, sans-serif; }
  #map { width: 100%; height: __MAP_H__px; }
  #controls {
    display: flex; align-items: center; gap: 12px;
    padding: 10px 6px; height: __CTRL_H__px; box-sizing: border-box;
  }
  #controls button {
    width: 34px; height: 34px; border: 1px solid #ccc; border-radius: 6px;
    background: #fff; cursor: pointer; font-size: 14px;
  }
  #controls button:disabled { opacity: 0.4; cursor: default; }
  #slider { flex: 1; }
  #slider:disabled { opacity: 0.4; }
  #label {
    font-variant-numeric: tabular-nums; min-width: 96px; text-align: right;
    color: #333;
  }
  /* Show albedo cells as sharp blocks instead of bilinear-smoothed mush. */
  .albedo-overlay {
    image-rendering: -webkit-optimize-contrast;
    image-rendering: crisp-edges;
    image-rendering: pixelated;
  }
  .albedo-legend {
    background: rgba(255, 255, 255, 0.9);
    padding: 6px 8px; border-radius: 4px;
    font: 11px/1.2 -apple-system, system-ui, sans-serif; color: #333;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.3);
  }
  .albedo-legend .title { font-weight: 600; text-align: center; margin-bottom: 4px; }
  .albedo-legend .body { display: flex; gap: 4px; }
  .albedo-legend .bar {
    width: 14px; height: 110px; border: 1px solid #999;
  }
  .albedo-legend .ticks {
    display: flex; flex-direction: column; justify-content: space-between;
    height: 112px; font-variant-numeric: tabular-nums;
  }
</style>
<div id="map"></div>
<div id="controls">
  <button id="play" title="Play / pause">&#9654;</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1"/>
  <span id="label"></span>
</div>
<script>
(function () {
  var D = __PAYLOAD__;
  var map = L.map('map');
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxNativeZoom: 19, maxZoom: 22,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map);
  L.control.scale({ position: 'bottomleft', metric: true, imperial: true }).addTo(map);

  if (D.bounds) { map.fitBounds(D.bounds); }
  else { map.setView(D.alaska.center, D.alaska.zoom); }
  setTimeout(function () { map.invalidateSize(); }, 120);

  var frames = D.frames || [];

  if (D.legend && frames.length) {
    var legend = L.control({ position: 'bottomright' });
    legend.onAdd = function () {
      var div = L.DomUtil.create('div', 'albedo-legend');
      var ticks = D.legend.ticks.map(function (t) {
        return '<span>' + t + '</span>';
      }).join('');
      div.innerHTML =
        '<div class="title">' + D.legend.title + '</div>' +
        '<div class="body">' +
          '<div class="bar" style="background:' + D.legend.gradient + '"></div>' +
          '<div class="ticks">' + ticks + '</div>' +
        '</div>';
      return div;
    };
    legend.addTo(map);
  }

  if (D.outline) {
    L.geoJSON(D.outline, {
      style: { color: '#1f77b4', weight: 2, fillOpacity: 0.1 },
      onEachFeature: function (feat, layer) {
        var p = feat.properties || {};
        layer.bindTooltip(
          '<b>' + (p.glacier_name || '') + '</b><br>RGI ID: ' + (p.rgi_id || ''),
          { sticky: true }
        );
      }
    }).addTo(map);
  }

  var slider = document.getElementById('slider');
  var label = document.getElementById('label');
  var playBtn = document.getElementById('play');
  var overlay = null;
  var cur = 0;
  var timer = null;

  function render(i) {
    cur = i;
    var fr = frames[i];
    if (!overlay) {
      overlay = L.imageOverlay(fr.url, D.bounds, {
        opacity: D.opacity, interactive: false, className: 'albedo-overlay'
      }).addTo(map);
    } else {
      overlay.setUrl(fr.url);
    }
    label.textContent = fr.label;
    if (+slider.value !== i) slider.value = i;
  }

  if (!frames.length) {
    slider.disabled = true;
    playBtn.disabled = true;
    label.textContent = 'Select a glacier';
    return;
  }

  frames.forEach(function (f) { var im = new Image(); im.src = f.url; });  // preload
  slider.max = frames.length - 1;
  render(frames.length - 1);

  slider.addEventListener('input', function (e) { render(+e.target.value); });
  playBtn.addEventListener('click', function () {
    if (timer) {
      clearInterval(timer); timer = null; playBtn.innerHTML = '&#9654;';
      return;
    }
    playBtn.innerHTML = '&#10073;&#10073;';
    timer = setInterval(function () {
      render((cur + 1) % frames.length);
    }, 350);
  });
})();
</script>
""".replace("__MAP_H__", str(MAP_HEIGHT)).replace("__CTRL_H__", str(CONTROLS_HEIGHT))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Alaska Glacier Albedo Explorer", layout="wide")

gdf = load_glacier_data()
gdf = gdf.dropna(subset=["rgi_id"]).drop_duplicates(subset=["rgi_id"])
_name_by_rgi = dict(zip(gdf["rgi_id"], gdf["glacier_name"].fillna("")))


def glacier_name_of(rgi_id: str) -> str:
    """Display name for a glacier, falling back to its RGI ID when unnamed."""
    return (_name_by_rgi.get(rgi_id) or "").strip() or rgi_id


def glacier_label(rgi_id: str) -> str:
    """Dropdown label: ``Name — RGI ID`` (or just the RGI ID when unnamed)."""
    name = (_name_by_rgi.get(rgi_id) or "").strip()
    return f"{name} — {rgi_id}" if name else rgi_id


rgi_options = sorted(
    gdf["rgi_id"],
    key=lambda r: (not (_name_by_rgi.get(r) or "").strip(), glacier_label(r).lower()),
)  # named glaciers first (A-Z), then unnamed by RGI ID

st.title("Alaska Glacier Albedo Explorer")
st.markdown(
    "Explore albedo data for glaciers across Alaska. Pick a glacier in the sidebar, "
    "then scrub or play the time slider under the map."
)

with st.sidebar:
    st.header("Controls")
    selected_rgi = st.selectbox(
        "Search by glacier name or RGI ID:",
        options=rgi_options,
        index=None,
        format_func=glacier_label,
        placeholder="Type a glacier name or RGI ID...",
        key="glacier_search",
    )

frame_data = None
outline_geojson = None
if selected_rgi:
    match = gdf.loc[gdf["rgi_id"] == selected_rgi]
    row = match.iloc[0]
    frame_data = build_frames(row["gcs_uri"], selected_rgi)
    outline_geojson = json.loads(match.to_json())

with st.sidebar:
    st.divider()
    st.subheader("Export")
    if not selected_rgi:
        st.markdown("*Select a glacier to enable downloads.*")
    else:
        name = glacier_name_of(selected_rgi)
        try:
            st.link_button(
                f"Download {name} NetCDF",
                signed_download_url(row["cube_uri"]),
                help="Temporary link straight from Google Cloud Storage (expires in 15 min).",
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not create download link: {exc}")

if frame_data and "error" in frame_data:
    st.error(f"Failed to load data: {frame_data['error']}")

render_map(frame_data if frame_data and "error" not in frame_data else None, outline_geojson)
