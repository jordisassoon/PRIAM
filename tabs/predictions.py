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
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import your models and loader
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WAPLS
from models.rf import RF
from utils.dataloader import ProxyDataLoader
from validation.cross_validate import run_grouped_cv
from utils.colors import color_map


@st.cache_data
def plot_mat_tsne(
    modern_coords, fossil_coords, train_metadata, test_metadata, predictions, neighbors_info, target
):
    combined_df, links_df = create_mat_tsne_df(
        modern_coords, fossil_coords, train_metadata, test_metadata, predictions, neighbors_info, target
    )

    offset = 2
    tsne1_min = combined_df["TSNE1"].min() - offset
    tsne1_max = combined_df["TSNE1"].max() + offset
    tsne2_min = combined_df["TSNE2"].min() - offset
    tsne2_max = combined_df["TSNE2"].max() + offset

    # Selection for fossils
    fossil_select = alt.selection_single(fields=["OBSNAME"], on="click")

    # Base scatter: Fossil + Modern
    base = (
        alt.Chart(combined_df)
        .mark_circle()
        .encode(
            x=alt.X(
                "TSNE1:Q",
                title="t-SNE 1",
                scale=alt.Scale(domain=(tsne1_min, tsne1_max)),
            ),
            y=alt.Y(
                "TSNE2:Q",
                title="t-SNE 2",
                scale=alt.Scale(domain=(tsne2_min, tsne2_max)),
            ),
            color=alt.Color(
                "PlotType:N",
                scale=alt.Scale(
                    domain=["Fossil", "Modern", "Neighbor"],
                    range=["red", "steelblue", "orange"],
                ),
                legend=alt.Legend(title="Point Type"),
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
                scale=alt.Scale(
                    domain=["Fossil", "Modern", "Neighbor"],
                    range=["red", "steelblue", "orange"],
                ),
                legend=alt.Legend(title="Point Type"),
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

    return chart


@st.cache_data
def create_mat_tsne_df(
    modern_coords, fossil_coords, train_metadata, test_metadata, predictions, neighbors_info, target
):
    # Prepare modern dataframe
    modern_df = train_metadata.copy()
    modern_df["Type"] = "Modern"
    modern_df["Predicted"] = np.nan
    modern_df["TSNE1"] = modern_coords[:, 0]
    modern_df["TSNE2"] = modern_coords[:, 1]

    # Prepare fossil dataframe
    fossil_df = pd.DataFrame(
        {
            "OBSNAME": test_metadata["OBSNAME"],
            "TSNE1": fossil_coords[:, 0],
            "TSNE2": fossil_coords[:, 1],
            "Type": "Fossil",
            "Predicted": predictions,
        }
    )

    combined_df = pd.concat([modern_df, fossil_df], ignore_index=True)

    # Build links dataframe
    link_rows = []
    for i, info in enumerate(neighbors_info):
        fossil_name = test_metadata.iloc[i]["OBSNAME"]  # TODO: use actual labels
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

    # Define PlotType for fossils/modern
    combined_df["PlotType"] = combined_df["Type"].apply(
        lambda t: "Fossil" if t == "Fossil" else "Modern"
    )
    links_df["PlotType"] = "Neighbor"

    return combined_df, links_df


@st.cache_data
def compute_mat_tsne(X_train, X_test):
    """Compute t-SNE coordinates for MAT nearest neighbors visualization."""
    combined = np.vstack([X_train, X_test])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    coords = tsne.fit_transform(combined)

    modern_coords = coords[: len(X_train)]
    fossil_coords = coords[len(X_train) :]

    return modern_coords, fossil_coords


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
            brt_model = model

    axis_string = f"{axis}"

    # --- Combine Predictions ---
    df_preds = pd.DataFrame({axis_string: ages_or_depths.values})
    for name, preds in predictions_dict.items():
        df_preds[f"{name}"] = preds
    df_plot = df_preds.set_index(axis_string)

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

    # --- Prepare Plotly figure ---
    fig = go.Figure()

    for col in df_plot_combined.columns:
        if col == axis_string:
            continue

        # Determine base model name for color
        base_name = col.replace("_smoothed", "")
        color = color_map.get(base_name, "#7f7f7f")  # default gray if not in map

        # Determine line style and name
        if "_smoothed" in col:
            line_width = 3
            dash = "solid"
            name = f"{base_name} (Smoothed)"
        else:
            line_width = 1
            dash = "dot"
            name = f"{base_name} (Per Sample)"

        fig.add_trace(
            go.Scatter(
                x=df_plot_combined[axis_string],
                y=df_plot_combined[col],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=line_width, dash=dash),
                hovertemplate=f"%{{x}} {axis_string}<br>%{{y}} Prediction<br>Model: {name}<extra></extra>",
            )
        )

    # --- Toggle for mirroring X axis ---
    mirror_x = st.checkbox(f"Mirror {axis} axis", value=False)

    # --- Layout ---
    fig.update_layout(
        width=900,
        height=500,
        margin=dict(l=60, r=20, t=50, b=80),
        xaxis_title=axis_string,
        yaxis_title="Prediction",
        xaxis=dict(autorange="reversed" if mirror_x else True),  # ✅ Simpler
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)

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

        modern_coords, fossil_coords = compute_mat_tsne(X_train_aligned, X_test_aligned)

        # Load training metadata for tooltips
        train_climate_file.seek(0)
        train_metadata = pd.read_csv(train_climate_file, encoding="latin1")
        test_metadata = pd.DataFrame({"OBSNAME": [f"{axis}: {val}" for val in ages_or_depths]})

        neighbors_info = mat_model.get_neighbors_info(X_test_aligned.values, 
            metadata_df=train_metadata, return_distance=True
        )
        
        chart = plot_mat_tsne(
            modern_coords=modern_coords,
            fossil_coords=fossil_coords,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            predictions=predictions_dict["MAT"],
            neighbors_info=neighbors_info,
            target=target,
        )

        st.altair_chart(chart, use_container_width=True)

        st.info(
            "💡 This t-SNE projection shows assemblage composition space. "
            "Click a red fossil point to highlight its nearest modern analogues (orange) and connecting lines. "
            "Other points fade out."
        )

    # --- RF Tree Visualization ---
    if model_choice in ["RF", "All"]:
        if model_choice == "All":
            rf_model = model
        else:
            rf_model = model

        st.subheader(f"RF Model Visualization")

        # --- Feature Importances ---
        if hasattr(rf_model, "feature_importances_"):
            st.markdown("### 🔍 Feature Importance")

            importances = rf_model.feature_importances_
            feature_names = list(X_train_aligned.columns)

            # Create DataFrame
            importance_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

            # --- Plotly Bar Chart ---
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (Mean Decrease in Impurity)",
                color="Importance",
                color_continuous_scale="viridis",
                height=600,
            )

            fig.update_layout(
                yaxis=dict(autorange="reversed"),  # highest importance on top
                margin=dict(l=100, r=20, t=50, b=50),
            )

            st.plotly_chart(fig, use_container_width=True)

    # --- BRT Tree Visualization ---
    if model_choice in ["BRT", "All"]:
        if model_choice == "All":
            brt_model = model
        else:
            brt_model = model

        st.subheader(f"BRT Model Visualization")

        # --- Feature Importances ---
        if hasattr(brt_model, "feature_importances_"):
            st.markdown("### 🔍 Feature Importance")

            # For LightGBM, you can choose 'split' or 'gain'
            importances = (
                brt_model.feature_importances_
            )  # default is 'split', use brt_model.feature_importance(importance_type='gain') if needed

            feature_names = list(X_train_aligned.columns)

            # Create DataFrame
            importance_df = (
                pd.DataFrame({"Feature": feature_names, "Importance": importances})
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )

            # --- Plotly Bar Chart ---
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance (LightGBM)",
                color="Importance",
                color_continuous_scale="viridis",
                height=600,
            )

            fig.update_layout(
                yaxis=dict(autorange="reversed"),  # highest importance on top
                margin=dict(l=100, r=20, t=50, b=50),
            )

            st.plotly_chart(fig, use_container_width=True)
