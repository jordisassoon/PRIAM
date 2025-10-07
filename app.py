import streamlit as st
import pandas as pd
import numpy as np
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WA_PLS
from models.rf import RF
from utils.dataloader import PollenDataLoader
from utils.cross_validate import run_grouped_cv
from scipy.ndimage import gaussian_filter1d
import altair as alt

import streamlit as st
import pandas as pd
import folium
import branca.colormap as cm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# === function (adapted from your script, no click needed) ===
def generate_map(pollen_file, coords_file, sample_id_col="OBSNAME",
                 lat_col="LATI", lon_col="LONG", alt_col="ALTI",
                 output_html="map.html"):

    pollen_file.seek(0)
    pollen_df = pd.read_csv(pollen_file, delimiter=',', encoding="latin1")
    
    coords_file.seek(0)
    coords_df = pd.read_csv(coords_file, delimiter=',', encoding="latin1")
    merged_df = pd.merge(pollen_df, coords_df, on=sample_id_col, how="inner")

    if not merged_df.empty:
        center_lat = merged_df[lat_col].mean()
        center_lon = merged_df[lon_col].mean()
    else:
        center_lat, center_lon = 0, 0

    m = folium.Map(location=[center_lat, center_lon], zoom_start=4)

    if alt_col in merged_df.columns:
        min_alt, max_alt = merged_df[alt_col].min(), merged_df[alt_col].max()
        colormap = cm.linear.viridis.scale(min_alt, max_alt)
        colormap.caption = "Altitude"
        colormap.add_to(m)
    else:
        colormap = lambda x: "blue"

    for _, row in merged_df.iterrows():
        altitude = row.get(alt_col, None)
        color = colormap(altitude) if altitude is not None else "blue"
        popup_text = f"{sample_id_col}: {row[sample_id_col]}<br>{alt_col}: {altitude if altitude is not None else 'N/A'}"
        folium.CircleMarker(
            location=[row[lat_col], row[lon_col]],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=popup_text
        ).add_to(m)

    m.save(output_html)
    return output_html

# App title
st.title("🌿 Pollen-based Climate Reconstruction")

# Sidebar inputs
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["MAT", "BRT", "RF", "All"],  # Add "All" option
    index=0
)
target = st.sidebar.selectbox(
    "Target climate variable",
    ["TANN", "Temp_season", "MTWA", "MTCO", "PANN", "Temp_wet", "Temp_dry", "P_wet", "P_dry", "P_season"]
)
k = st.sidebar.slider("Number of neighbors (MAT only)", 1, 20, 5)
cv_folds = st.sidebar.slider("Cross-validation folds", 1, 10, 1)
# pls_components = st.sidebar.slider("PLS components (WA-PLS only)", 1, 10, 3)

random_seed = st.sidebar.number_input("Random seed", value=42)

# File uploads
st.sidebar.header("Upload Data")
train_climate_file = st.sidebar.file_uploader("Training Climate CSV", type=["csv"])
train_pollen_file = st.sidebar.file_uploader("Training Pollen CSV", type=["csv"])
test_pollen_file = st.sidebar.file_uploader("Test Fossil Pollen CSV", type=["csv"])
taxa_mask_file = st.sidebar.file_uploader("Taxa mask CSV", type=["csv"])
coords_file = st.sidebar.file_uploader("Coordinates file (CSV)", type=["csv"])

if train_climate_file and train_pollen_file and test_pollen_file:
    # Load data
    loader = PollenDataLoader(
        climate_file=train_climate_file,
        pollen_file=train_pollen_file,
        test_file=test_pollen_file,
        mask_file=taxa_mask_file,
    )

    X_train, y_train, obs_names = loader.load_training_data(target=target)
    X_test, ages = loader.load_test_data()
    X_train_aligned, X_test_aligned, shared_cols = loader.align_taxa(X_train, X_test)

    # Prepare models
    available_models = {
        "MAT": (MAT, {"k": k}),
        "BRT": (BRT, {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 3, "random_state": random_seed}),
        # "WA-PLS": (WA_PLS, {"n_components": pls_components}),
        "RF": (RF, {"n_estimators": 100, "max_depth": 6, "random_state": random_seed})
    }

    # Determine which models to run
    if model_choice == "All":
        models_to_run = available_models
    else:
        models_to_run = {model_choice: available_models[model_choice]}

    predictions_dict = {}
    metrics_display = []

    for name, (model_class, params) in models_to_run.items():
        model = model_class(**params)

        # Cross-validation
        if cv_folds > 1:
            st.write(f"📊 Running {cv_folds}-fold cross-validation for {name}...")
            scores_rmse, scores_r2 = run_grouped_cv(
                model_class, params,
                X_train_aligned, y_train, obs_names,
                n_splits=cv_folds, seed=random_seed, loader=loader
            )

            rmse_mean, rmse_std = np.mean(scores_rmse), np.std(scores_rmse)
            r2_mean, r2_std = np.mean(scores_r2), np.std(scores_r2)

            metrics_display.append({
                "Model": name,
                "RMSE": f"{rmse_mean:.3f} ± {rmse_std:.3f}",
                "R²": f"{r2_mean:.3f} ± {r2_std:.3f}"
            })

        # Train on full dataset + predict fossils
        model.fit(X_train_aligned, y_train)
        predictions_dict[name] = model.predict(X_test_aligned)
        
        if name == "MAT":
            mat_model = model

    # Combine predictions into dataframe
    df_preds = pd.DataFrame({"Age": ages.values})
    for name, preds in predictions_dict.items():
        df_preds[f"{name}_{target}"] = preds

    # Set Age as index
    df_plot = df_preds.set_index("Age")
    
    # Sidebar smoothing control
    smoothing_sigma = st.slider(
        "Gaussian smoothing (σ)", 
        min_value=0.0, 
        max_value=10.0, 
        value=2.0, 
        step=0.1
    )

    # Apply Gaussian smoothing to all prediction columns
    if smoothing_sigma > 0:
        smoothed_df = df_plot.apply(lambda col: gaussian_filter1d(col, sigma=smoothing_sigma))
        smoothed_df = smoothed_df.add_suffix("_smoothed")
        df_plot_combined = pd.concat([df_plot, smoothed_df], axis=1).reset_index()
    else:
        df_plot_combined = df_plot.reset_index()

    # Melt dataframe for Altair
    df_melted = df_plot_combined.melt(id_vars="Age", var_name="Model", value_name="Prediction")

    # Define line thickness
    df_melted["Thickness"] = df_melted["Model"].apply(lambda x: 4 if "_smoothed" in x else 1)

    # Altair chart
    chart = (
        alt.Chart(df_melted)
        .mark_line()
        .encode(
            x=alt.X("Age", title="Age"),
            y=alt.Y("Prediction", title=target, scale=alt.Scale(zero=False)),  # dynamic y
            color="Model",
            strokeWidth="Thickness"
        )
        .interactive()
    )

    # Display chart in Streamlit
    st.altair_chart(chart, use_container_width=True)

    # Display CV metrics in nice format
    if metrics_display:
        st.subheader("📊 Cross-validation Metrics")
        metrics_df = pd.DataFrame(metrics_display).set_index("Model")
        st.table(metrics_df)

    # Display predictions
    st.subheader("Predictions")
    st.write(df_preds)

    # Download
    csv = df_preds.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download predictions as CSV",
        data=csv,
        file_name="predictions.csv",
        mime="text/csv",
    )
    
    # --- Taxa selection expander ---
    if shared_cols is not None and len(shared_cols) > 0:
        with st.expander("Select taxa to include in the model"):
            # Dictionary to hold user selections
            taxa_selection = {}
            for taxa in shared_cols:
                taxa_selection[taxa] = st.checkbox(taxa, value=True)  # default checked

            # Filter columns based on selections
            selected_taxa = [taxa for taxa, include in taxa_selection.items() if include]

            if len(selected_taxa) == 0:
                st.warning("⚠️ No taxa selected. Predictions may fail.")
            else:
                # Apply selection to aligned data
                X_train_aligned = X_train_aligned[selected_taxa]
                X_test_aligned = X_test_aligned[selected_taxa]
    
    # === MAT Interactive Nearest Neighbors (t-SNE Space) ===
    if model_choice == "MAT" or model_choice == "All":
        if model_choice == "All":
            mat_model = mat_model
        else:
            mat_model = model
        
        st.subheader("🎯 MAT Nearest Neighbors Explorer (t-SNE Space)")

        from sklearn.manifold import TSNE

        # Fit t-SNE on combined taxa data
        combined_matrix = np.vstack([X_train_aligned.values, X_test_aligned.values])
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=random_seed)
        coords = tsne.fit_transform(combined_matrix)

        # Split coordinates
        modern_coords = coords[:len(X_train_aligned)]
        fossil_coords = coords[len(X_train_aligned):]

        # Prepare modern dataframe
        train_climate_file.seek(0)
        train_meta = pd.read_csv(train_climate_file, encoding="latin1")
        modern_df = train_meta.copy()
        modern_df["Type"] = "Modern"
        modern_df["Predicted"] = np.nan
        modern_df["TSNE1"] = modern_coords[:, 0]
        modern_df["TSNE2"] = modern_coords[:, 1]

        # Prepare fossil dataframe
        fossil_df = pd.DataFrame({
            "OBSNAME": [f"Fossil_{i}" for i in range(len(fossil_coords))],
            "TSNE1": fossil_coords[:, 0],
            "TSNE2": fossil_coords[:, 1],
            "Type": "Fossil",
            "Age": ages.values,
            f"Predicted_{target}": predictions_dict["MAT"]
        })
        fossil_df["Predicted"] = fossil_df[f"Predicted_{target}"]

        # Get nearest neighbor info from MAT
        neighbor_info = mat_model.get_neighbors_info(X_test_aligned, train_meta, return_distance=True)

        # Build links dataframe
        link_rows = []
        for i, info in enumerate(neighbor_info):
            fossil_name = f"Fossil_{i}"
            f_tsne1, f_tsne2 = fossil_coords[i]
            for n in info["neighbors"]:
                obsname = n["metadata"]["OBSNAME"]
                if obsname in modern_df["OBSNAME"].values:
                    m_row = modern_df.loc[modern_df["OBSNAME"] == obsname].iloc[0]
                    link_rows.append({
                        "fossil": fossil_name,
                        "neighbor": obsname,
                        "distance": n["distance"],
                        "fossil_TSNE1": f_tsne1,
                        "fossil_TSNE2": f_tsne2,
                        "modern_TSNE1": m_row["TSNE1"],
                        "modern_TSNE2": m_row["TSNE2"]
                    })
        links_df = pd.DataFrame(link_rows)
        links_df = links_df.rename(columns={"fossil": "OBSNAME"})

        # Combine modern + fossil for base chart
        combined_df = pd.concat([modern_df, fossil_df], ignore_index=True)

        # Compute explicit TSNE axis limits
        tsne1_min, tsne1_max = combined_df["TSNE1"].min(), combined_df["TSNE1"].max()
        tsne2_min, tsne2_max = combined_df["TSNE2"].min(), combined_df["TSNE2"].max()

        # Altair selection
        fossil_select = alt.selection_point(fields=["OBSNAME"], on="click")

        # Base scatter
        base = alt.Chart(combined_df).mark_circle().encode(
            x=alt.X("TSNE1:Q", scale=alt.Scale(domain=(tsne1_min, tsne1_max))),
            y=alt.Y("TSNE2:Q", scale=alt.Scale(domain=(tsne2_min, tsne2_max))),
            color=alt.condition(
                alt.datum.Type == "Fossil",
                alt.value("red"),
                alt.value("steelblue")
            ),
            opacity=alt.condition(fossil_select, alt.value(1.0), alt.value(0.3)),
            tooltip=["OBSNAME:N", "Type:N", "Predicted:Q"]
        ).add_params(fossil_select)

        # Highlight neighbors
        neighbor_highlight = (
            alt.Chart(links_df)
            .mark_circle(size=120, color="orange")
            .encode(
                x="modern_TSNE1:Q",
                y="modern_TSNE2:Q",
                tooltip=["neighbor:N", "distance:Q"]
            )
            .transform_filter(fossil_select)
        )

        # Connections
        connections = (
            alt.Chart(links_df)
            .mark_line(color="orange", opacity=0.6)
            .encode(
                x="fossil_TSNE1:Q",
                y="fossil_TSNE2:Q",
                x2="modern_TSNE1:Q",
                y2="modern_TSNE2:Q",
                tooltip=["neighbor:N", "distance:Q"]
            )
            .transform_filter(fossil_select)
        )

        # Combine charts using alt.layer()
        chart = alt.layer(base, neighbor_highlight, connections).resolve_scale(x='shared', y='shared').interactive()

        st.altair_chart(chart, use_container_width=True)

        st.info(
            "💡 This t-SNE projection shows assemblage composition space. "
            "Click a red fossil point to highlight its nearest modern analogues (orange) and connecting lines. "
            "Other points fade out."
        )
    
    if train_pollen_file and coords_file:
        output_html = "map_output.html"
        map_path = generate_map(train_pollen_file, coords_file, output_html=output_html)

        # Make clickable link (open in new tab)
        map_url = Path(map_path).resolve().as_uri()
        st.markdown(f"[Open Interactive Map]({map_url})", unsafe_allow_html=True)
        with open(map_path, "r", encoding="utf-8") as f:
            folium_html = f.read()
        st.components.v1.html(folium_html, height=500, scrolling=True)
        
        with open(map_path, "rb") as f:
            btn = st.download_button(
                label="Download Map HTML",
                data=f,
                file_name="map_output.html",
                mime="text/html"
            )

    
    # === Taxa distribution per climate target (Altair) ===
    st.subheader("🌱 Taxa Preference per Climate Target")

    if train_climate_file and train_pollen_file and shared_cols is not None:
        # Select a taxa
        taxa_options = shared_cols
        selected_taxa = st.selectbox("Select taxa for distribution plot", taxa_options)

        # Select target climate variable
        target_options = ["TANN", "Temp_season", "MTWA", "MTCO", "PANN", "Temp_wet",
                        "Temp_dry", "P_wet", "P_dry", "P_season"]
        selected_target = st.selectbox("Select target climate variable", target_options)

        # Number of bins
        bins = st.slider("Number of bins for target variable", min_value=3, max_value=50, value=25)

        # Load pollen and climate CSVs
        train_pollen_file.seek(0)
        pollen_df = pd.read_csv(train_pollen_file, delimiter=",", encoding="latin1")
        train_climate_file.seek(0)
        climate_df = pd.read_csv(train_climate_file, delimiter=",", encoding="latin1")

        # Merge datasets
        merged_df = pd.merge(pollen_df, climate_df, on="OBSNAME", how="inner")

        if selected_taxa not in merged_df.columns:
            st.error(f"Taxa '{selected_taxa}' not found in pollen dataset")
        else:
            # Bin target variable
            merged_df['binned_target'] = pd.cut(merged_df[selected_target], bins=bins)

            # Calculate taxa preference per bin
            taxa_sum = merged_df.groupby('binned_target')[selected_taxa].sum()
            total_count = merged_df.groupby('binned_target')[selected_taxa].count()
            preference = (taxa_sum / total_count).reset_index()
            preference.rename(columns={selected_taxa: 'preference'}, inplace=True)
            preference['bin_label'] = preference['binned_target'].apply(lambda x: f"{x.left:.2f}–{x.right:.2f}")

            chart = (
                alt.Chart(preference)
                .mark_bar()
                .encode(
                    x=alt.X('bin_label:N', title=selected_target, sort=None),
                    y=alt.Y('preference:Q', title=f'{selected_taxa} Average Count'),
                    tooltip=[alt.Tooltip('bin_label:N', title='Target bin'),
                            alt.Tooltip('preference:Q', title=f'{selected_taxa} average')]
                )
            )
            st.altair_chart(chart, use_container_width=True)
    
    # === Climate Variables Scatter Plot ===
    st.subheader("🌡️ Climate Variables Scatter Plot")

    if train_climate_file:
        # Select X and Y variables
        climate_options = ["TANN", "Temp_season", "MTWA", "MTCO", "PANN",
                        "Temp_wet", "Temp_dry", "P_wet", "P_dry", "P_season"]
        col1, col2 = st.columns(2)
        with col1:
            x_var = st.selectbox("X-axis climate variable", climate_options, key="x_var_only")
        with col2:
            y_var = st.selectbox("Y-axis climate variable", [x for x in climate_options if x != x_var], key="y_var_only")

        # Load climate CSV
        train_climate_file.seek(0)
        climate_df = pd.read_csv(train_climate_file, delimiter=",", encoding="latin1")

        if x_var not in climate_df.columns or y_var not in climate_df.columns:
            st.error("Selected climate variables not found in the dataset.")
        else:
            # Combine observed and predicted for legend
            obs_df = climate_df.copy()
            obs_df['Type'] = 'Modern'
            obs_df['Fossil (Predicted)'] = np.nan
            obs_df['Age'] = np.nan

            pred_col = [col for col in df_preds.columns if "_smoothed" not in col and col != "Age"][0]
            pred_df = df_preds.copy()
            pred_df['Type'] = 'Fossil (Predicted)'
            pred_df['OBSNAME'] = np.nan
            pred_df['Fossil (Predicted)'] = pred_df[pred_col]
            pred_df['Age'] = pred_df['Age']

            # Ensure X/Y values exist
            pred_df[x_var] = climate_df[x_var].values[:len(pred_df)]
            pred_df[y_var] = climate_df[y_var].values[:len(pred_df)]

            combined_df = pd.concat([obs_df[[x_var, y_var, 'OBSNAME','Age','Fossil (Predicted)','Type']],
                                    pred_df[[x_var, y_var, 'OBSNAME','Age','Fossil (Predicted)','Type']]], ignore_index=True)

            # Determine min/max for dynamic scaling
            x_min, x_max = combined_df[x_var].min(), combined_df[x_var].max()
            y_min, y_max = combined_df[y_var].min(), combined_df[y_var].max()

            # Plot
            scatter_chart = (
                alt.Chart(combined_df)
                .mark_circle(size=60, opacity=0.7)
                .encode(
                    x=alt.X(f"{x_var}:Q", title=x_var, scale=alt.Scale(domain=(x_min, x_max))),
                    y=alt.Y(f"{y_var}:Q", title=y_var, scale=alt.Scale(domain=(y_min, y_max))),
                    color=alt.Color("Type:N", title="Data Type", scale=alt.Scale(domain=['Modern','Fossil (Predicted)'], range=['blue','red'])),
                    tooltip=[
                        alt.Tooltip('Type:N'),
                        alt.Tooltip('OBSNAME:N'),
                        alt.Tooltip('Age:Q'),
                        alt.Tooltip('Fossil (Predicted):Q'),
                        alt.Tooltip(f"{x_var}:Q"),
                        alt.Tooltip(f"{y_var}:Q")
                    ]
                )
                .interactive()
            )

            st.altair_chart(scatter_chart, use_container_width=True)

else:
    st.info("Please upload all three files to continue.")
