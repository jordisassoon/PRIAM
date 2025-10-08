import streamlit as st
import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter1d
import altair as alt
import matplotlib.pyplot as plt
from utils.map_utils import generate_map
from sklearn.manifold import TSNE
from sklearn.tree import plot_tree
import tempfile

# Import your models and loader
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WAPLS
from models.rf import RF
from utils.dataloader import PollenDataLoader
from validation.cross_validate import run_grouped_cv

def show_tab(train_climate_file, train_pollen_file, test_pollen_file,
             taxa_mask_file, model_choice, target,
             n_neighbors, brt_trees, rf_trees, cv_folds, random_seed):

    st.header("📊 Predictions & Model Visualizations")

    if not (train_climate_file and train_pollen_file and test_pollen_file):
        st.info("Please upload all required files to run predictions.")
        return

    # --- Load Data ---
    loader = PollenDataLoader(
        climate_file=train_climate_file,
        pollen_file=train_pollen_file,
        test_file=test_pollen_file,
        mask_file=taxa_mask_file
    )

    X_train, y_train, obs_names = loader.load_training_data(target)
    X_test, ages = loader.load_test_data()
    X_train_aligned, X_test_aligned, shared_cols = loader.align_taxa(X_train, X_test)

    # --- Prepare Models ---
    available_models = {
        "MAT": (MAT, {"n_neighbors": n_neighbors}),
        "BRT": (BRT, {"n_estimators": brt_trees, "learning_rate": 0.05, "max_depth": 6, "random_state": random_seed}),
        # "WAPLS": (WAPLS, {"n_components": 5, "weighted": True}),
        "RF": (RF, {"n_estimators": rf_trees, "max_depth": 6, "random_state": random_seed})
    }

    if model_choice == "All":
        models_to_run = available_models
    else:
        models_to_run = {model_choice: available_models[model_choice]}

    predictions_dict = {}
    metrics_display = []

    # --- Run Models ---
    for name, (model_class, params) in models_to_run.items():
        model = model_class(**params)

        # Cross-validation
        if cv_folds > 1:
            st.write(f"📊 Running {cv_folds}-fold CV for {name}...")
            scores = run_grouped_cv(
                model_class, params,
                X_train_aligned, y_train, obs_names,
                n_splits=cv_folds, seed=random_seed, loader=loader
            )
            metrics_display.append({
                "Model": name,
                "RMSE": f"{np.mean(scores['rmse']):.3f} ± {np.std(scores['rmse']):.3f}",
                "R²": f"{np.mean(scores['r2']):.3f} ± {np.std(scores['r2']):.3f}"
            })

        # Train on full dataset
        model.fit(X_train_aligned, y_train)
        predictions_dict[name] = model.predict(X_test_aligned)
        if name == "MAT":
            mat_model = model
        if name == "RF":
            rf_model = model

    # --- Combine Predictions ---
    df_preds = pd.DataFrame({"Age": ages.values})
    for name, preds in predictions_dict.items():
        df_preds[f"{name}_{target}"] = preds
    df_plot = df_preds.set_index("Age")

    # --- Gaussian Smoothing ---
    smoothing_sigma = st.slider("Gaussian smoothing (σ)", 0.0, 10.0, 2.0, 0.1)
    if smoothing_sigma > 0:
        smoothed_df = df_plot.apply(lambda col: gaussian_filter1d(col, sigma=smoothing_sigma))
        smoothed_df = smoothed_df.add_suffix("_smoothed")
        df_plot_combined = pd.concat([df_plot, smoothed_df], axis=1).reset_index()
    else:
        df_plot_combined = df_plot.reset_index()

    # --- Altair Line Chart ---
    df_melted = df_plot_combined.melt(id_vars="Age", var_name="Model", value_name="Prediction")
    df_melted["Thickness"] = df_melted["Model"].apply(lambda x: 4 if "_smoothed" in x else 1)

    chart = (
        alt.Chart(df_melted)
        .mark_line()
        .encode(
            x="Age",
            y=alt.Y("Prediction", scale=alt.Scale(zero=False)),
            color="Model",
            strokeWidth="Thickness"
        )
        .interactive()
    )
    st.altair_chart(chart, use_container_width=True)

    # --- Cross-validation Metrics ---
    if metrics_display:
        st.subheader("📊 Cross-validation Metrics")
        st.table(pd.DataFrame(metrics_display).set_index("Model"))

    # --- Download Predictions ---
    st.download_button(
        "Download Predictions CSV",
        df_preds.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv"
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

    # --- RF/BRT Tree Visualization ---
    if model_choice in ["RF", "All"]:
        if model_choice == "All":
            rf_model = model
        else:
            rf_model = model
        st.subheader(f"🌳 RF Tree Visualization")
        if hasattr(rf_model, "estimators_"):
            num_trees = len(rf_model.estimators_)
            tree_idx = st.slider(f"Select tree index for RF", 0, num_trees - 1, 0)
            tree_to_plot = rf_model.estimators_[tree_idx]

            viz_choice = st.radio("Visualization type", ["Simple (matplotlib)", "Detailed (dtreeviz)"])
            if viz_choice == "Simple (matplotlib)":
                fig, ax = plt.subplots(figsize=(20, 10))
                plot_tree(tree_to_plot, filled=True, max_depth=3, fontsize=8)
                st.pyplot(fig)
            elif viz_choice == "Detailed (dtreeviz)":
                try:
                    from dtreeviz.trees import dtreeviz
                    viz_model = dtreeviz(
                        tree_to_plot,
                        X_train_aligned,
                        y_train,
                        target_name=target,
                        feature_names=list(X_train_aligned.columns)
                    )
                    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tmp:
                        tmp_path = tmp.name
                        viz_model.save(tmp_path)
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=700, scrolling=True)
                except Exception as e:
                    st.error(f"dtreeviz failed: {e}")
    
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
