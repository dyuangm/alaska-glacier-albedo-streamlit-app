import streamlit as st
import folium
from streamlit_folium import st_folium
import datetime
import geopandas as gpd
from shapely.geometry import Polygon, MultiPolygon

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
        min_value=datetime.date(2010, 1, 1),
        max_value=datetime.date(2023, 12, 31),
        value=datetime.date(2023, 7, 15),
        format="YYYY-MM-DD"
    )
    
    st.divider()
    
    st.subheader("Export")
    st.markdown("*Select a glacier to enable downloads.*")
    if selected_glacier:
        gcs_uri = gdf[gdf['glacier_name'] == selected_glacier].iloc[0]['gcs_uri']
        st.download_button(
            label=f"Download {selected_glacier} NetCDF",
            data=b"mock_netcdf_bytes",
            file_name=f"{selected_glacier.replace(' ', '_')}_{selected_date}.nc",
            mime="application/x-netcdf",
            disabled=True,
            help=f"Will eventually download from: {gcs_uri}"
        )
    else:
        st.markdown("*Select a glacier to enable downloads.*")
        
m = folium.Map(location=[64.2, -149.5], zoom_start=5, tiles="CartoDB Voyager")

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
    returned_objects=[]
)

if st.session_state.move_camera:
    st.session_state.move_camera = False