import pandas as pd
import folium
from folium.plugins import Fullscreen
import branca.colormap as cm
import streamlit as st

def generate_map(
    pollen_file,
    coords_file,
    sample_id_col="OBSNAME",
    lat_col="LATI",
    lon_col="LONG",
    alt_col="ALTI",
    output_html="map.html",
    topo=False  # toggle topographic map
):
    # --- Read CSVs ---
    
    pollen_df = pd.read_csv(pollen_file, delimiter=',', encoding="latin1")
    coords_df = pd.read_csv(coords_file, delimiter=',', encoding="latin1")
    merged_df = pd.merge(pollen_df, coords_df, on=sample_id_col, how="inner")

    # --- Map center ---
    center_lat = merged_df[lat_col].mean() if not merged_df.empty else 0
    center_lon = merged_df[lon_col].mean() if not merged_df.empty else 0

    # --- Create empty map ---
    m = folium.Map(location=[center_lat, center_lon], zoom_start=4, tiles=None)

    # --- Add default OpenStreetMap layer ---
    folium.TileLayer(
        "OpenStreetMap",
        name="OpenStreetMap / Normal",
        control=True,
        show=not topo
    ).add_to(m)

    # --- Add ESRI World Topographic Map layer ---
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles &copy; Esri &mdash; Esri, USGS, NOAA",
        name="ESRI Topographic",
        control=True,
        show=topo
    ).add_to(m)

    # --- Fullscreen plugin ---
    Fullscreen(position='topright').add_to(m)

    # --- Colormap for altitude ---
    if alt_col in merged_df.columns and not merged_df[alt_col].isnull().all():
        colormap = cm.linear.viridis.scale(merged_df[alt_col].min(), merged_df[alt_col].max())
        colormap.caption = "Altitude"
        colormap.add_to(m)
        get_color = lambda alt: colormap(alt) if pd.notnull(alt) else "blue"
    else:
        get_color = lambda alt: "blue"

    # --- Add markers ---
    for _, row in merged_df.iterrows():
        altitude = row.get(alt_col, None)
        color = get_color(altitude)
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{sample_id_col}: {row[sample_id_col]}<br>{alt_col}: {altitude if altitude else 'N/A'}"
        ).add_to(m)

    # --- Layer control ---
    folium.LayerControl().add_to(m)

    # --- Save map ---
    m.save(output_html)
    return output_html
