import pandas as pd
import folium
import click
import branca.colormap as cm


@click.command()
@click.argument("pollen_file", type=click.Path(exists=True))
@click.argument("coords_file", type=click.Path(exists=True))
@click.option("--sample-id-col", default="OBSNAME", help="Column name for sample IDs")
@click.option("--lat-col", default="LATI", help="Column name for latitude")
@click.option("--lon-col", default="LONG", help="Column name for longitude")
@click.option("--alt-col", default="ALTI", help="Column name for altitude")
@click.option(
    "--output-html",
    default="map.html",
    help="Output HTML file for interactive map",
)
def main(
    pollen_file,
    coords_file,
    sample_id_col,
    lat_col,
    lon_col,
    alt_col,
    output_html,
):
    """Load pollen and coordinate CSVs, merge them, and plot locations on an interactive map with altitude color scale."""

    # ==== LOAD DATA ====
    pollen_df = pd.read_csv(pollen_file, delimiter=",", encoding="latin1")
    coords_df = pd.read_csv(coords_file, delimiter=",", encoding="latin1")

    # pollen_df = pollen_df.rename(columns={"ï»¿OBSNAME": "OBSNAME"})

    # ==== MERGE DATASETS ====
    merged_df = pd.merge(pollen_df, coords_df, on=sample_id_col, how="inner")

    # ==== CREATE MAP ====
    if not merged_df.empty:
        center_lat = merged_df[lat_col].mean()
        center_lon = merged_df[lon_col].mean()
    else:
        center_lat, center_lon = 0, 0

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=4,
    )

    # ==== COLOR SCALE ====
    if alt_col in merged_df.columns:
        min_alt, max_alt = merged_df[alt_col].min(), merged_df[alt_col].max()
        colormap = cm.linear.viridis.scale(min_alt, max_alt)
        colormap.caption = "Altitude"
        colormap.add_to(m)
    else:
        colormap = lambda x: "blue"

    print(merged_df[merged_df["OBSNAME"] == "PATAM18_CT2"])

    # ==== ADD POINTS ====
    for _, row in merged_df.iterrows():
        altitude = row.get(alt_col, None)
        color = colormap(altitude) if altitude is not None else "blue"
        popup_text = (
            f"{sample_id_col}: {row[sample_id_col]}<br>{alt_col}: {altitude if altitude is not None else 'N/A'}"
        )
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text,
        ).add_to(m)

    # ==== SAVE MAP ====
    m.save(output_html)
    print(f"Interactive map with altitude color scale saved to {output_html}")


if __name__ == "__main__":
    main()
