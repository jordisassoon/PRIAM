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
def plot_prediction_lines(df_plot_combined, axis_string, mirror_x):
    # --- Prepare Plotly figure for predictions ---
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
            )
        )

    # --- Layout ---
    fig.update_layout(
        width=800,
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis_title=axis_string,
        yaxis_title="Prediction",
        xaxis=dict(autorange="reversed" if mirror_x else True),  # ✅ Simpler
        hovermode="x unified",
    )

    st.plotly_chart(fig, use_container_width=True)


def feature_importance_plot(model, model_name, X_train):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_ / model.feature_importances_.sum()
        importances = importances * 100  # convert to percentage
        feature_names = list(X_train.columns)

        # Create DataFrame
        importance_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
        importance_df["Importance"] = importance_df["Importance"].round(2)  # round for better display
        importance_df = importance_df[importance_df["Importance"] > 0][:25]  # top 25 features

        # --- Plotly Bar Chart ---
        fig = px.bar(
            importance_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title=f"{model_name}",
            color="Importance",
            color_continuous_scale="viridis",
            height=600,
        )

        fig.update_layout(
            yaxis=dict(autorange="reversed"),  # highest importance on top
            margin=dict(l=100, r=20, t=50, b=50),
            coloraxis_showscale=False,  # hide colorbar if you want cleaner look
        )

        st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def plot_mat_tsne(combined_df, links_df):
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
                "Type:N",
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
                "Type:N",
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
    chart = alt.layer(base, neighbor_points, connections).resolve_scale(x="shared", y="shared").interactive()

    st.altair_chart(chart, use_container_width=True)


@st.cache_data
def create_mat_tsne_df(
    tsne_coords,
    train_metadata,
    test_metadata,
    ground_truth,
    predictions,
    neighbors_info,
    target,
):
    # Prepare modern dataframe
    tsne_df = pd.DataFrame(tsne_coords, columns=["TSNE1", "TSNE2"])
    tsne_df["Type"] = ["Modern"] * len(train_metadata) + ["Fossil"] * len(test_metadata)
    tsne_df["OBSNAME"] = list(train_metadata["OBSNAME"]) + list(
        test_metadata[st.session_state.get("prediction_axis", "Age")]
    )
    tsne_df["Predicted"] = list(ground_truth) + list(predictions)

    # Build links dataframe
    link_rows = []
    for i, info in enumerate(neighbors_info):
        fossil_name = test_metadata.iloc[i][st.session_state.get("prediction_axis", "Age")]
        f_tsne1, f_tsne2 = tsne_df.loc[tsne_df["OBSNAME"] == fossil_name, ["TSNE1", "TSNE2"]].values[0]
        for n in info["neighbors"]:
            obsname = n["metadata"]["OBSNAME"]
            if obsname in tsne_df["OBSNAME"].values:
                m_row = tsne_df.loc[tsne_df["OBSNAME"] == obsname].iloc[0]
                link_rows.append(
                    {
                        "OBSNAME": fossil_name,
                        "neighbor": obsname,
                        "distance": n["distance"],
                        "fossil_TSNE1": f_tsne1,
                        "fossil_TSNE2": f_tsne2,
                        "modern_TSNE1": m_row["TSNE1"],
                        "modern_TSNE2": m_row["TSNE2"],
                        "Type": "Neighbor",
                    }
                )

    links_df = pd.DataFrame(link_rows)

    return tsne_df, links_df


@st.cache_data
def compute_mat_tsne(X_train, X_test):
    """Compute t-SNE coordinates for MAT nearest neighbors visualization."""
    combined = np.vstack([X_train, X_test])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_coords = tsne.fit_transform(combined)

    return tsne_coords


@st.cache_data
def cached_fit_mat(X_train, y_train, n_neighbors):
    mat_model = MAT(n_neighbors=n_neighbors)
    mat_model.fit(X_train, y_train)
    return mat_model


@st.cache_data
def cached_fit_rf(X_train, y_train, X_test, n_trees, max_depth, random_seed):
    rf_model = RF(n_estimators=n_trees, max_depth=max_depth, random_state=random_seed)
    rf_model.fit(X_train, y_train)
    return rf_model


@st.cache_data
def cached_fit_brt(X_train, y_train, n_trees, learning_rate, max_depth, random_seed):
    brt_model = BRT(
        n_estimators=n_trees,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_seed,
    )
    brt_model.fit(X_train, y_train)
    return brt_model


def show_tab(
    X_train,
    X_test,
    y_train,
    train_metadata,
    test_metadata,
):
    st.header("Predictions & Model Visualizations")

    if X_train is None or X_test is None or y_train is None:
        st.warning("Please upload training and test datasets in the 'Data Upload' section.")
        return

    predictions_dict = {}

    if st.session_state.get("use_mat"):
        mat_model = cached_fit_mat(X_train, y_train, st.session_state.get("n_neighbors", None))
        predictions_dict["MAT"] = mat_model.predict(X_test)
    if st.session_state.get("use_rf"):
        rf_model = cached_fit_rf(
            X_train,
            y_train,
            X_test,
            st.session_state.get("rf_trees", None),
            st.session_state.get("rf_max_depth", None),
            st.session_state.get("random_seed", None),
        )
        predictions_dict["RF"] = rf_model.predict(X_test)
    if st.session_state.get("use_brt"):
        brt_model = cached_fit_brt(
            X_train,
            y_train,
            st.session_state.get("brt_trees", None),
            st.session_state.get("brt_learning_rate", None),
            st.session_state.get("brt_max_depth", None),
            st.session_state.get("random_seed", None),
        )
        predictions_dict["BRT"] = brt_model.predict(X_test)

    # --- Combine Predictions ---
    df_preds = pd.DataFrame(predictions_dict)
    df_preds = df_preds.join(test_metadata.reset_index(drop=True))  # Join metadata for display
    axis_string = st.session_state.get("prediction_axis", "Age")
    df_plot = df_preds.set_index(axis_string).sort_index()

    # --- Layout for Plot and Controls ---
    predictions_plot_expander = st.expander("Prediction Plot Settings", expanded=False)

    with predictions_plot_expander:
        # --- Gaussian Smoothing ---
        smoothing_sigma = st.slider("Gaussian smoothing (σ)", 0.0, 10.0, 2.0, 0.1, key="smoothing_sigma")
        if smoothing_sigma > 0:
            smoothed_df = df_plot.apply(lambda col: gaussian_filter1d(col, sigma=smoothing_sigma))
            smoothed_df = smoothed_df.add_suffix("_smoothed")
            df_plot_combined = pd.concat([df_plot, smoothed_df], axis=1).reset_index()
        else:
            df_plot_combined = df_plot.reset_index()

        # --- Mirror X Axis ---
        mirror_x = st.toggle(f"Mirror x-axis", value=False, key="mirror_x")

    # --- Plot Predictions ---
    plot_prediction_lines(df_plot_combined, axis_string, mirror_x)

    # --- Show DataFrame in Streamlit ---
    st.subheader("Prediction Data Table")
    st.dataframe(df_plot)  # 👈 Interactive table

    # === MAT Interactive Nearest Neighbors (t-SNE Space) ===
    if st.session_state.get("use_mat"):
        st.subheader("MAT Nearest Neighbors Explorer (t-SNE Space)")

        tsne_coords = compute_mat_tsne(X_train, X_test)

        neighbors_info = mat_model.get_neighbors_info(
            X_test.values,
            metadata_df=train_metadata,
            return_distance=True,
        )

        combined_df, links_df = create_mat_tsne_df(
            tsne_coords=tsne_coords,
            train_metadata=train_metadata,
            test_metadata=test_metadata,
            ground_truth=y_train,
            predictions=predictions_dict["MAT"],
            neighbors_info=neighbors_info,
            target=st.session_state["target"],
        )

        plot_mat_tsne(combined_df, links_df)

        st.info(
            "This t-SNE projection shows assemblage composition space. "
            "Click a red fossil point to highlight its nearest modern analogues (orange) and connecting lines. "
            "Other points fade out."
        )

    # --- RF Tree Visualization ---
    if st.session_state.get("use_rf") and st.session_state.get("use_brt"):
        st.subheader(f"Trees Feature Importance Visualization")
        col1, col2 = st.columns(2)
        with col1:
            feature_importance_plot(rf_model, "Random Forest", X_train)
        with col2:
            feature_importance_plot(brt_model, "Boosted Regression Trees", X_train)

    elif st.session_state.get("use_rf"):
        st.subheader(f"Trees Feature Importance Visualization")
        feature_importance_plot(rf_model, "Random Forest", X_train)
    elif st.session_state.get("use_brt"):
        st.subheader(f"Trees Feature Importance Visualization")
        feature_importance_plot(brt_model, "Boosted Regression Trees", X_train)

    if st.session_state.get("use_rf") or st.session_state.get("use_brt"):
        st.info(
            "Feature importance indicates how much each taxa contributed to the model's predictions. "
            "Higher importance means the taxa was more influential."
        )
