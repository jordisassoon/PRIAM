import pandas as pd
import folium
import branca.colormap as cm

def generate_map(pollen_file, coords_file, sample_id_col="OBSNAME",
                 lat_col="LATI", lon_col="LONG", alt_col="ALTI",
                 output_html="map.html"):
    pollen_file.seek(0)
    pollen_df = pd.read_csv(pollen_file, delimiter=',', encoding="latin1")
    coords_file.seek(0)
    coords_df = pd.read_csv(coords_file, delimiter=',', encoding="latin1")
    merged_df = pd.merge(pollen_df, coords_df, on=sample_id_col, how="inner")

    center_lat = merged_df[lat_col].mean() if not merged_df.empty else 0
    center_lon = merged_df[lon_col].mean() if not merged_df.empty else 0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
    if alt_col in merged_df.columns:
        colormap = cm.linear.viridis.scale(merged_df[alt_col].min(), merged_df[alt_col].max())
        colormap.caption = "Altitude"
        colormap.add_to(m)
    else:
        colormap = lambda x: "blue"

    for _, row in merged_df.iterrows():
        altitude = row.get(alt_col, None)
        color = colormap(altitude) if altitude else "blue"
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{sample_id_col}: {row[sample_id_col]}<br>{alt_col}: {altitude if altitude else 'N/A'}"
        ).add_to(m)

    m.save(output_html)
    return output_html
