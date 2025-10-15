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
from utils.dataloader import ProxyDataLoader
from validation.cross_validate import run_grouped_cv


def show_tab(
    train_climate_file,
    train_proxy_file,
    test_proxy_file,
    taxa_mask_file,
    model_choice,
    target,
    n_neighbors,
    brt_trees,
    rf_trees,
    cv_folds,
    random_seed,
    axis,
):

    st.header("Predictions & Model Visualizations")

    if not (train_climate_file and train_proxy_file and test_proxy_file):
        st.warning("Please upload all required files to run predictions.")
        return

    try:
        train_climate_file.seek(0)
    except:
        st.warning("Please upload the training climate dataset.")
        return
    try:
        train_proxy_file.seek(0)
    except:
        st.warning("Please upload the training proxy dataset.")
        return
    try:
        test_proxy_file.seek(0)
    except:
        st.warning("Please upload the test proxy dataset.")
        return

    # --- Load Data ---
    loader = ProxyDataLoader(
        climate_file=train_climate_file,
        proxy_file=train_proxy_file,
        test_file=test_proxy_file,
        mask_file=taxa_mask_file,
    )

    X_train, y_train, obs_names = loader.load_training_data(target)
    X_test, ages_or_depths = loader.load_test_data(age_or_depth=axis)
    X_train_aligned, X_test_aligned, shared_cols = loader.align_taxa(X_train, X_test)

    # --- Taxa selection expander ---
    if shared_cols is not None and len(shared_cols) > 0:
        with st.expander("Select taxa to include in the model"):
            # Dictionary to hold user selections
            taxa_selection = {}
            for taxa in shared_cols:
                taxa_selection[taxa] = st.checkbox(taxa, value=True)  # default checked

            # Filter columns based on selections
            selected_taxa = [
                taxa for taxa, include in taxa_selection.items() if include
            ]

            if len(selected_taxa) == 0:
                st.warning("⚠️ No taxa selected. Predictions may fail.")
            else:
                # Apply selection to aligned data
                X_train_aligned = X_train_aligned[selected_taxa]
                X_test_aligned = X_test_aligned[selected_taxa]

    # --- Prepare Models ---
    available_models = {
        "MAT": (MAT, {"n_neighbors": n_neighbors}),
        "BRT": (
            BRT,
            {
                "n_estimators": brt_trees,
                "learning_rate": 0.05,
                "max_depth": 6,
                "random_state": random_seed,
            },
        ),
        # "WAPLS": (WAPLS, {"n_components": 5, "weighted": True}),
        "RF": (
            RF,
            {"n_estimators": rf_trees, "max_depth": 6, "random_state": random_seed},
        ),
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
            st.write(f"Running {cv_folds}-fold CV for {name}...")
            scores = run_grouped_cv(
                model_class,
                params,
                X_train_aligned,
                y_train,
                obs_names,
                n_splits=cv_folds,
                seed=random_seed,
                loader=loader,
            )
            metrics_display.append(
                {
                    "Model": name,
                    "RMSE": f"{np.mean(scores['rmse']):.3f} ± {np.std(scores['rmse']):.3f}",
                    "R²": f"{np.mean(scores['r2']):.3f} ± {np.std(scores['r2']):.3f}",
                }
            )

        # Train on full dataset
        model.fit(X_train_aligned, y_train)
        predictions_dict[name] = model.predict(X_test_aligned)
        if name == "MAT":
            mat_model = model
        if name == "RF":
            rf_model = model

    # --- Combine Predictions ---
    df_preds = pd.DataFrame({f"{axis}": ages_or_depths.values})
    for name, preds in predictions_dict.items():
        df_preds[f"{name}"] = preds
    df_plot = df_preds.set_index(f"{axis}")

    # --- Gaussian Smoothing ---
    smoothing_sigma = st.slider("Gaussian smoothing (σ)", 0.0, 10.0, 2.0, 0.1)
    if smoothing_sigma > 0:
        smoothed_df = df_plot.apply(
            lambda col: gaussian_filter1d(col, sigma=smoothing_sigma)
        )
        smoothed_df = smoothed_df.add_suffix("_smoothed")
        df_plot_combined = pd.concat([df_plot, smoothed_df], axis=1).reset_index()
    else:
        df_plot_combined = df_plot.reset_index()

    # --- Altair Line Chart ---
    df_melted = df_plot_combined.melt(
        id_vars=f"{axis}", var_name="Model", value_name="Prediction"
    )
    df_melted["Thickness"] = df_melted["Model"].apply(
        lambda x: 4 if "_smoothed" in x else 1
    )

    # --- Toggle for mirroring X axis ---
    mirror_x = st.checkbox(f"Mirror {axis} axis", value=False)

    # --- Build Altair chart with optional mirrored X ---
    x_scale = alt.Scale(zero=False, reverse=mirror_x)
    
    df_melted["Type"] = df_melted["Model"].apply(
        lambda x: f"{x.replace('_smoothed', '')} (Smoothed)" if "_smoothed" in x else f"{x} (Per Sample)"
    )

    # Optionally, keep original Model for hover info
    chart = (
        alt.Chart(df_melted)
        .mark_line()
        .encode(
            x=alt.X(f"{axis}", scale=alt.Scale(zero=False, reverse=mirror_x)),
            y=alt.Y("Prediction", scale=alt.Scale(zero=False)),
            color=alt.Color("Type", title=""),  # This will show only "per sample" / "smoothed"
            strokeWidth=alt.StrokeWidth("Thickness", legend=None),
            tooltip=["Model", "Prediction"]  # optional: show exact model on hover
        )
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

    # --- Show DataFrame in Streamlit ---
    st.subheader("Prediction Data Table")
    st.dataframe(df_preds)  # 👈 Interactive table

    # === MAT Interactive Nearest Neighbors (t-SNE Space) ===
    if model_choice == "MAT" or model_choice == "All":
        if model_choice == "All":
            mat_model = mat_model
        else:
            mat_model = model

        st.subheader("MAT Nearest Neighbors Explorer (t-SNE Space)")

        from sklearn.manifold import TSNE

        # Fit t-SNE on combined taxa data
        combined_matrix = np.vstack([X_train_aligned.values, X_test_aligned.values])
        tsne = TSNE(
            n_components=2, perplexity=30, learning_rate=200, random_state=42
        )
        coords = tsne.fit_transform(combined_matrix)

        # Split coordinates
        modern_coords = coords[: len(X_train_aligned)]
        fossil_coords = coords[len(X_train_aligned) :]

        # Prepare modern dataframe
        train_climate_file.seek(0)
        train_meta = pd.read_csv(train_climate_file, encoding="latin1")
        modern_df = train_meta.copy()
        modern_df["Type"] = "Modern"
        modern_df["Predicted"] = np.nan
        modern_df["TSNE1"] = modern_coords[:, 0]
        modern_df["TSNE2"] = modern_coords[:, 1]

        # Prepare fossil dataframe with formatted labels
        if axis == "Age":
            fossil_labels = ages_or_depths.apply(lambda x: f"Age: {x}").astype(str)
        elif axis == "Depth":
            fossil_labels = ages_or_depths.apply(lambda x: f"Depth: {x}").astype(str)
        else:
            fossil_labels = [f"Fossil_{i}" for i in range(len(ages_or_depths))]

        # Prepare fossil dataframe
        fossil_df = pd.DataFrame(
            {
                "OBSNAME": fossil_labels,
                "TSNE1": fossil_coords[:, 0],
                "TSNE2": fossil_coords[:, 1],
                "Type": "Fossil",
                f"{axis}": ages_or_depths.values,
                f"Predicted_{target}": predictions_dict["MAT"],
            }
        )
        fossil_df["Predicted"] = fossil_df[f"Predicted_{target}"]

        # Get nearest neighbor info from MAT
        neighbor_info = mat_model.get_neighbors_info(
            X_test_aligned, train_meta, return_distance=True
        )

        # Build links dataframe
        link_rows = []
        for i, info in enumerate(neighbor_info):
            fossil_name = fossil_df.iloc[i]["OBSNAME"]
            f_tsne1, f_tsne2 = fossil_coords[i]
            for n in info["neighbors"]:
                obsname = n["metadata"]["OBSNAME"]
                if obsname in modern_df["OBSNAME"].values:
                    m_row = modern_df.loc[modern_df["OBSNAME"] == obsname].iloc[0]
                    link_rows.append(
                        {
                            "fossil": fossil_name,
                            "neighbor": obsname,
                            "distance": n["distance"],
                            "fossil_TSNE1": f_tsne1,
                            "fossil_TSNE2": f_tsne2,
                            "modern_TSNE1": m_row["TSNE1"],
                            "modern_TSNE2": m_row["TSNE2"],
                        }
                    )
        links_df = pd.DataFrame(link_rows)
        links_df = links_df.rename(columns={"fossil": "OBSNAME"})

        # Combine modern + fossil for base chart
        combined_df = pd.concat([modern_df, fossil_df], ignore_index=True)

        # Compute explicit TSNE axis limits
        tsne1_min, tsne1_max = combined_df["TSNE1"].min(), combined_df["TSNE1"].max()
        tsne2_min, tsne2_max = combined_df["TSNE2"].min(), combined_df["TSNE2"].max()

        # Altair selection
        fossil_select = alt.selection_point(fields=["OBSNAME"], on="click")

        # Define PlotType for fossils/modern
        combined_df["PlotType"] = combined_df["Type"].apply(lambda t: "Fossil" if t == "Fossil" else "Modern")
        links_df["PlotType"] = "Neighbor"

        # Base scatter: Fossil + Modern
        base = (
            alt.Chart(combined_df)
            .mark_circle()
            .encode(
                x=alt.X("TSNE1:Q", title="t-SNE 1", scale=alt.Scale(domain=(tsne1_min, tsne1_max))),
                y=alt.Y("TSNE2:Q", title="t-SNE 2", scale=alt.Scale(domain=(tsne2_min, tsne2_max))),
                color=alt.Color(
                    "PlotType:N",
                    scale=alt.Scale(domain=["Fossil", "Modern", "Neighbor"], range=["red", "steelblue", "orange"]),
                    legend=alt.Legend(title="Point Type")
                ),
                opacity=alt.condition(fossil_select, alt.value(1.0), alt.value(0.3)),
                tooltip=["OBSNAME:N", "Type:N", "Predicted:Q"],
            )
            .add_params(fossil_select)
        )

        # Neighbor points: orange, only show when fossil is selected
        neighbor_points = (
            alt.Chart(links_df)
            .mark_circle()
            .encode(
                x="modern_TSNE1:Q",
                y="modern_TSNE2:Q",
                color=alt.Color(
                    "PlotType:N",
                    scale=alt.Scale(domain=["Fossil", "Modern", "Neighbor"], range=["red", "steelblue", "orange"]),
                    legend=alt.Legend(title="Point Type")
                ),
                tooltip=["neighbor:N", "distance:Q"],
                opacity=alt.condition(fossil_select, alt.value(1.0), alt.value(0.0)),
            )
        )

        # Connections: lines between fossils and neighbors
        connections = (
            alt.Chart(links_df)
            .mark_line(color="orange", opacity=0.6)
            .encode(
                x="fossil_TSNE1:Q",
                y="fossil_TSNE2:Q",
                x2="modern_TSNE1:Q",
                y2="modern_TSNE2:Q",
                tooltip=["neighbor:N", "distance:Q"],
            )
            .transform_filter(fossil_select)
        )

        # Combine layers
        chart = (
            alt.layer(base, neighbor_points, connections)
            .resolve_scale(x="shared", y="shared")
            .interactive()
        )

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
        st.subheader(f"RF Tree Visualization")
        if hasattr(rf_model, "estimators_"):
            num_trees = len(rf_model.estimators_)
            tree_idx = st.slider(f"Select tree index for RF", 0, num_trees - 1, 0)
            tree_to_plot = rf_model.estimators_[tree_idx]

            viz_choice = st.radio(
                "Visualization type", ["Simple (matplotlib)", "Detailed (dtreeviz)"]
            )
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
                        feature_names=list(X_train_aligned.columns),
                    )
                    with tempfile.NamedTemporaryFile(
                        suffix=".svg", delete=False
                    ) as tmp:
                        tmp_path = tmp.name
                        viz_model.save(tmp_path)
                    with open(tmp_path, "r", encoding="utf-8") as f:
                        st.components.v1.html(f.read(), height=700, scrolling=True)
                except Exception as e:
                    st.error(f"dtreeviz failed: {e}")
