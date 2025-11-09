import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import altair as alt
from scipy.spatial import distance
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.neighbors import KernelDensity
import plotly.express as px
import shutil

from utils.map_utils import generate_map

# ======================
# 🔹 CACHING UTILITIES
# ======================


@st.cache_data
def hellinger_transform(df):
    """Apply Hellinger transformation to the DataFrame."""
    df = df + 1e-9  # Avoid division by zero
    return df.apply(lambda x: np.sqrt(x) / np.sqrt(x.sum()), axis=1)


@st.cache_data
def load_csv(file):
    """Read a CSV file safely with caching."""
    file.seek(0)
    return pd.read_csv(file, encoding="latin1")


@st.cache_data
def normalize_rows(df):
    """Normalize rows to sum to 1."""
    row_sums = df.sum(axis=1)
    return df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)


@st.cache_data
def compute_mmd(X_train, X_test, gamma=1.0):
    """Compute Maximum Mean Discrepancy (MMD) between train/test."""

    # Hellinger transformation
    X_train = hellinger_transform(pd.DataFrame(X_train)).values
    X_test = hellinger_transform(pd.DataFrame(X_test)).values

    XX = rbf_kernel(X_train, X_train, gamma)
    YY = rbf_kernel(X_test, X_test, gamma)
    XY = rbf_kernel(X_train, X_test, gamma)
    return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)


@st.cache_data
def compute_pca_kde(X_train, X_test, bandwidth=0.5):
    """Compute PCA + KDE density probabilities for test samples."""

    # Apply Hellinger transformation
    X_train = hellinger_transform(pd.DataFrame(X_train)).values
    X_test = hellinger_transform(pd.DataFrame(X_test)).values

    pca = PCA(n_components=X_test.shape[1] // 5, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X_train_pca)
    log_probs = kde.score_samples(X_test_pca)
    probs = np.exp(log_probs)
    probs_norm = (probs - probs.min()) / (probs.max() - probs.min())
    return probs_norm


@st.cache_data
def compute_embeddings(X_train, X_test):
    """Compute PCA, t-SNE, and UMAP embeddings for visualization."""
    combined = np.vstack([X_train, X_test])
    labels = ["Train"] * len(X_train) + ["Test"] * len(X_test)

    # Hellinger transformation
    h_combined = hellinger_transform(pd.DataFrame(combined)).values

    # PCA
    pca = PCA(n_components=2)
    pca_emb = pca.fit_transform(h_combined)

    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    tsne_emb = tsne.fit_transform(combined)

    return pca_emb, tsne_emb, labels


@st.cache_data
def cached_generate_map(train_proxy_file, coords_file, topo):
    new_topo = topo
    """Cache the map generation process."""
    output_html = "map_output.html"
    return generate_map(
        train_proxy_file, coords_file, output_html=output_html, topo=new_topo
    )


# ======================
# 🔹 MAIN APP FUNCTION
# ======================


def show_tab(train_climate_file, train_proxy_file, test_proxy_file, coords_file, axis):
    st.header("Data Exploration: Distribution & Train–Test Comparison")

    if not train_climate_file:
        st.warning(
            "To begin exploring data, please upload the training climate dataset."
        )
        return

    # === Load climate data ===
    climate_df = load_csv(train_climate_file)

    # === Climate Variables Scatter Plot ===
    st.subheader("Modern Climate Variables Scatter Plot")

    climate_options = climate_df.drop(
        ["OBSNAME"], axis=1, errors="ignore"
    ).columns.tolist()
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox(
            "X-axis climate variable", climate_options, key="x_var_only", index=0
        )
    with col2:
        y_var = st.selectbox(
            "Y-axis climate variable", climate_options, key="y_var_only", index=1
        )

    if x_var not in climate_df.columns or y_var not in climate_df.columns:
        st.error("Selected climate variables not found in the dataset.")
    else:
        obs_df = climate_df.copy()
        obs_df["Type"] = "Modern"
        x_min, x_max = obs_df[x_var].min(), obs_df[x_var].max()
        y_min, y_max = obs_df[y_var].min(), obs_df[y_var].max()

        scatter_chart = (
            alt.Chart(obs_df)
            .mark_circle(size=60, opacity=0.7)
            .encode(
                x=alt.X(
                    f"{x_var}:Q", title=x_var, scale=alt.Scale(domain=(x_min, x_max))
                ),
                y=alt.Y(
                    f"{y_var}:Q", title=y_var, scale=alt.Scale(domain=(y_min, y_max))
                ),
                tooltip=[
                    alt.Tooltip("Type:N"),
                    alt.Tooltip("OBSNAME:N"),
                    alt.Tooltip("Age:Q"),
                    alt.Tooltip(f"{x_var}:Q"),
                    alt.Tooltip(f"{y_var}:Q"),
                ],
            )
            .interactive()
        )
        st.altair_chart(scatter_chart, use_container_width=True)

    # === Training Proxy File ===
    if not train_proxy_file:
        st.warning(
            "To plot taxa distributions and compare train–test distributions, please upload the training proxy dataset."
        )
        return

    train_df = load_csv(train_proxy_file)

    # Determine labels based on the flag
    train_labels = train_df["OBSNAME"].astype(str)

    # === Taxa Preference Plot ===
    st.subheader("Taxa Preference per Climate Target")
    selected_target = st.selectbox("Select target climate variable", climate_options)

    taxa_list = [c for c in train_df.columns if c != "OBSNAME"]
    selected_taxa = st.selectbox("Select taxa for distribution plot", taxa_list)
    bins = st.slider("Number of bins for target variable", 1, 500, 25)

    merged_df = pd.merge(train_df, climate_df, on="OBSNAME", how="inner")
    merged_df["binned_target"] = pd.cut(merged_df[selected_target], bins=bins)
    taxa_sum = merged_df.groupby("binned_target")[selected_taxa].sum()
    total_count = merged_df.groupby("binned_target")[selected_taxa].count()
    preference = (taxa_sum / total_count).reset_index()
    preference.rename(columns={selected_taxa: "preference"}, inplace=True)
    preference["bin_label"] = preference["binned_target"].apply(
        lambda x: f"{x.left:.2f}–{x.right:.2f}"
    )

    chart = (
        alt.Chart(preference)
        .mark_bar()
        .encode(
            x=alt.X("bin_label:N", title=selected_target),
            y=alt.Y("preference:Q", title=f"{selected_taxa} Average Count"),
            tooltip=[alt.Tooltip("bin_label:N"), alt.Tooltip("preference:Q")],
        )
    )
    st.altair_chart(chart, use_container_width=True)

    # === Train vs Test Distribution Comparison ===
    if not test_proxy_file:
        st.warning(
            "To compare your train and test proxies, please upload the test proxy dataset."
        )
    else:
        test_df = load_csv(test_proxy_file)

        if axis == "Age":
            test_labels = test_df["Age"].apply(lambda x: f"Age: {x}").astype(str)
        elif axis == "Depth":
            test_labels = test_df["Depth"].apply(lambda x: f"Depth: {x}").astype(str)
        else:
            test_labels = test_df.index.astype(str)

        labels = pd.concat([train_labels, test_labels], ignore_index=True)

        shared_cols = [
            c for c in train_df.columns if c in test_df.columns and c != "OBSNAME"
        ]
        X_train = normalize_rows(train_df[shared_cols])
        X_test = normalize_rows(test_df[shared_cols])

        # --- Compute Metrics ---
        mmd_value = compute_mmd(X_train, X_test, gamma=1.0 / X_train.shape[1])
        st.metric("MMD (RBF Kernel)", f"{mmd_value:.5f}")

        probs_norm = compute_pca_kde(X_train, X_test)
        st.metric("Mean likelihood (PCA-KDE)", f"{np.mean(probs_norm):.3f}")

        kde_df = pd.DataFrame(
            {"Test Sample": np.arange(len(probs_norm)), "Probability": probs_norm}
        )
        kde_df["Sample"] = test_labels
        fig = px.bar(
            kde_df,
            x="Test Sample",
            y="Probability",
            hover_name="Sample",
            title="Test Sample Likelihood (PCA-KDE)",
        )
        fig.update_layout(
            yaxis_title="Normalized Probability", xaxis_title="Test Sample Index"
        )
        st.plotly_chart(fig, use_container_width=True)

        # --- Embeddings ---
        st.subheader("Low-Dimensional Embeddings (Train vs Test)")
        pca_emb, tsne_emb, set_labels = compute_embeddings(X_train, X_test)

        # PCA Plot
        pca_df = pd.DataFrame(pca_emb, columns=["PC1", "PC2"])
        pca_df["Sample"] = labels
        pca_df["Set"] = set_labels
        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Set",
            hover_name="Sample",
            title="PCA Projection",
            color_discrete_map={"Train": "steelblue", "Test": "red"},
        )
        st.plotly_chart(fig_pca, use_container_width=True)

        # t-SNE Plot
        tsne_df = pd.DataFrame(tsne_emb, columns=["Dim1", "Dim2"])
        tsne_df["Sample"] = labels
        tsne_df["Set"] = set_labels
        fig_tsne = px.scatter(
            tsne_df,
            x="Dim1",
            y="Dim2",
            color="Set",
            hover_name="Sample",
            title="t-SNE Projection",
            color_discrete_map={"Train": "steelblue", "Test": "red"},
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

    # === Coordinates Map ===
    if not coords_file:
        st.warning(
            "To plot your samples on a geographic map, please upload the training proxy coordinates dataset."
        )
        return

    st.subheader("Site Coordinates Map")
    map_path = cached_generate_map(train_proxy_file, coords_file, False)

    with open(map_path, "r", encoding="utf-8") as f:
        map_html = f.read()
    st.components.v1.html(map_html, height=800, scrolling=True)

    with open(map_path, "rb") as f:
        st.download_button(
            "Download Map HTML", f, file_name="map_output.html", mime="text/html"
        )
