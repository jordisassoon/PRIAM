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

def show_tab(train_climate_file, train_pollen_file, test_pollen_file, coords_file, axis):
    st.header("📈 Data Exploration: Distribution & Train–Test Comparison")

    # === TAXA DISTRIBUTION SECTION ===
    st.subheader("🌱 Taxa Preference per Climate Target")

    if train_climate_file and train_pollen_file:
        target_options = ["TANN", "Temp_season", "MTWA", "MTCO", "PANN", 
                          "Temp_wet", "Temp_dry", "P_wet", "P_dry", "P_season"]
        selected_target = st.selectbox("Select target climate variable", target_options)

        # Load data
        try:
            train_climate_file.seek(0)
        except:
            st.warning("Please upload the training climate dataset.")
        climate_df = pd.read_csv(train_climate_file, encoding="latin1")
        
        try:
            train_pollen_file.seek(0)
        except:
            st.warning("Please upload the training pollen dataset.")
        pollen_df = pd.read_csv(train_pollen_file, encoding="latin1")

        taxa_list = [c for c in pollen_df.columns if c != "OBSNAME"]
        selected_taxa = st.selectbox("Select taxa for distribution plot", taxa_list)
        bins = st.slider("Number of bins for target variable", 1, 500, 25)

        merged_df = pd.merge(pollen_df, climate_df, on="OBSNAME", how="inner")
        merged_df["binned_target"] = pd.cut(merged_df[selected_target], bins=bins)
        taxa_sum = merged_df.groupby("binned_target")[selected_taxa].sum()
        total_count = merged_df.groupby("binned_target")[selected_taxa].count()
        preference = (taxa_sum / total_count).reset_index()
        preference.rename(columns={selected_taxa: "preference"}, inplace=True)
        preference["bin_label"] = preference["binned_target"].apply(lambda x: f"{x.left:.2f}–{x.right:.2f}")

        chart = (
            alt.Chart(preference)
            .mark_bar()
            .encode(
                x=alt.X("bin_label:N", title=selected_target),
                y=alt.Y("preference:Q", title=f"{selected_taxa} Average Count"),
                tooltip=[alt.Tooltip("bin_label:N"), alt.Tooltip("preference:Q")]
            )
        )
        st.altair_chart(chart, use_container_width=True)

    # === TAXA DISTRIBUTION SECTION ===
    st.subheader("🌍 Site Coordinates Map")
    
    # --- Toggle for topographic map ---
    topo_toggle = st.checkbox("Show Topographic Map")

    # === MAP SECTION ===
    if train_pollen_file and coords_file:
        output_html = "map_output.html"

        # Generate the map (pass the tile type based on toggle)
        tile_type = "Stamen Terrain" if topo_toggle else "OpenStreetMap"
        map_path = generate_map(
            train_pollen_file,
            coords_file,
            output_html=output_html,
            topo=topo_toggle
        )

        # --- Embed in Streamlit ---
        with open(map_path, "r", encoding="utf-8") as f:
            map_html = f.read()

        st.components.v1.html(
            map_html,
            height=800,
            scrolling=True
        )

        # --- Download button ---
        with open(map_path, "rb") as f:
            st.download_button(
                "Download Map HTML",
                f,
                file_name="map_output.html",
                mime="text/html"
            )
    else:
        st.warning("Please upload the training pollen coordinates dataset.")
        
    # === TRAIN–TEST DISTRIBUTION COMPARISON ===
    if train_pollen_file and test_pollen_file:
        st.subheader("📊 Train vs Test Distribution Comparison")

        train_pollen_file.seek(0)
        train_df = pd.read_csv(train_pollen_file, encoding="latin1")
        test_pollen_file.seek(0)
        test_df = pd.read_csv(test_pollen_file, encoding="latin1")

        # Align columns
        shared_cols = [c for c in train_df.columns if c in test_df.columns and c != "OBSNAME"]
        X_train = train_df[shared_cols]
        X_test = test_df[shared_cols]

        # Assume X_train, X_test are pandas DataFrames from PollenDataLoader
        # Normalize rows as in your loader
        def normalize_rows(df):
            row_sums = df.sum(axis=1)
            return df.div(row_sums.replace(0, np.nan), axis=0).fillna(0)

        X_train = normalize_rows(X_train)
        X_test = normalize_rows(X_test)

        # --- MMD (Maximum Mean Discrepancy) ---
        def mmd(X, Y, gamma=1.0):
            XX = rbf_kernel(X, X, gamma)
            YY = rbf_kernel(Y, Y, gamma)
            XY = rbf_kernel(X, Y, gamma)
            return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)

        mmd_value = mmd(X_train, X_test, gamma=1.0 / X_train.shape[1])
        st.metric("MMD (RBF Kernel)", f"{mmd_value:.5f}")

        # --- Reduce dimensionality with PCA ---
        pca = PCA(n_components=10, random_state=42)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)

        # --- KDE in 10D ---
        kde = KernelDensity(kernel="gaussian", bandwidth=0.5).fit(X_train_pca)
        log_probs = kde.score_samples(X_test_pca)
        probs = np.exp(log_probs)  # convert log-density to density

        # Optional: normalize to 0–1 for interpretability
        probs_norm = (probs - probs.min()) / (probs.max() - probs.min())

        # Display a mean metric
        st.metric("Mean likelihood (PCA-KDE)", f"{np.mean(probs_norm):.3f}")

        # Plot per-sample probability as a bar chart
        kde_df = pd.DataFrame({"Test Sample": np.arange(len(probs_norm)), "Probability": probs_norm})
        fig = px.bar(kde_df, x="Test Sample", y="Probability", title="Test Sample Likelihood (PCA-KDE)")
        fig.update_layout(yaxis_title="Normalized Probability", xaxis_title="Test Sample Index")
        st.plotly_chart(fig, use_container_width=True)

        # --- Embedding visualizations ---
        st.subheader("🧭 Low-Dimensional Embeddings (Train vs Test)")
        combined = np.vstack([X_train, X_test])
        labels = ["Train"] * len(X_train) + ["Test"] * len(X_test)

        # PCA
        pca = PCA(n_components=2)
        pca_emb = pca.fit_transform(combined)
        pca_df = pd.DataFrame(pca_emb, columns=["PC1", "PC2"])
        pca_df["Set"] = labels
        fig_pca = px.scatter(pca_df, x="PC1", y="PC2", color="Set", title="PCA Projection")
        st.plotly_chart(fig_pca, use_container_width=True)

        # t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        tsne_emb = tsne.fit_transform(combined)
        tsne_df = pd.DataFrame(tsne_emb, columns=["Dim1", "Dim2"])
        tsne_df["Set"] = labels
        fig_tsne = px.scatter(tsne_df, x="Dim1", y="Dim2", color="Set", title="t-SNE Projection")
        st.plotly_chart(fig_tsne, use_container_width=True)

        # UMAP
        reducer = umap.UMAP(random_state=42)
        umap_emb = reducer.fit_transform(combined)
        umap_df = pd.DataFrame(umap_emb, columns=["UMAP1", "UMAP2"])
        umap_df["Set"] = labels
        fig_umap = px.scatter(umap_df, x="UMAP1", y="UMAP2", color="Set", title="UMAP Projection")
        st.plotly_chart(fig_umap, use_container_width=True)