import streamlit as st
import pandas as pd
import numpy as np
from utils.dataloader import PollenDataLoader
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WAPLS
from models.rf import RF
from validation.cross_validate import run_grouped_cv
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error

# Define a fixed color map for models
color_map = {
    "MAT": "#1f77b4",  # blue
    "BRT": "#ff7f0e",  # orange
    "RF": "#2ca02c"    # green
}

def show_tab(train_climate_file, train_pollen_file, test_pollen_file, taxa_mask_file,
             model_choice, target, n_neighbors, brt_trees, rf_trees, cv_folds, random_seed):

    st.header("🧪 Model Validation")
    st.info("Evaluate model performance using grouped cross-validation and diagnostic metrics (RMSE, MAE, R², r, KGE, Bias).")

    # --- Check inputs ---
    if not (train_climate_file and train_pollen_file):
        st.warning("Please upload both climate and pollen training datasets.")
        return

    # --- Load data ---
    loader = PollenDataLoader(
        climate_file=train_climate_file,
        pollen_file=train_pollen_file,
        test_file=test_pollen_file,
        mask_file=taxa_mask_file
    )

    X_train, y_train, obs_names = loader.load_training_data(target)

    # --- Available models ---
    available_models = {
        "MAT": (MAT, {"n_neighbors": n_neighbors}),
        "BRT": (BRT, {"n_estimators": brt_trees, "learning_rate": 0.05, "max_depth": 6, "random_state": random_seed}),
        "RF": (RF, {"n_estimators": rf_trees, "max_depth": 6, "random_state": random_seed})
    }

    if model_choice == "All":
        models_to_run = available_models
    else:
        models_to_run = {model_choice: available_models[model_choice]}

    metrics_table = []
    full_table = []
    error_metrics_list = []

    st.subheader(f"🔁 Running Cross-validation...")
    # --- Run CV for each model ---
    for name, (model_class, params) in models_to_run.items():

        with st.spinner(f"Running {cv_folds}-fold grouped CV on {name}..."):
            scores = run_grouped_cv(
                model_class, params,
                X_train, y_train, obs_names,
                n_splits=cv_folds, seed=random_seed, loader=loader
            )

        # Store means (numeric) for plotting
        metrics_table.append({
            "Model": name,
            "R²": np.mean(scores["r2"]),
            "r": np.mean(scores["r"]),
            "Spearman": np.mean(scores["spearman"]),
            "KGE": np.mean(scores["kge"]),
        })

        # Store mean ± std
        full_table.append({
            "Model": name,
            "RMSE": f"{np.mean(scores['rmse']):.2f} ± {np.std(scores['rmse']):.2f}",
            "MAE": f"{np.mean(scores['mae']):.2f} ± {np.std(scores['mae']):.2f}",
            "R²": f"{np.mean(scores['r2']):.2f} ± {np.std(scores['r2']):.2f}",
            "r": f"{np.mean(scores['r']):.2f} ± {np.std(scores['r']):.2f}",
            "Spearman": f"{np.mean(scores.get('spearman', [0])):.2f} ± {np.std(scores.get('spearman', [0])):.2f}",
            "KGE": f"{np.mean(scores['kge']):.2f} ± {np.std(scores['kge']):.2f}",
            "Bias": f"{np.mean(scores['bias']):.2f} ± {np.std(scores['bias']):.2f}"
        })

        # Store errors for histogram
        for rmse, mae, bias in zip(scores['rmse'], scores['mae'], scores['bias']):
            error_metrics_list.append({
                "Model": name,
                "RMSE": rmse,
                "MAE": mae,
                "Bias": bias
            })

    # --- Display full metrics table ---
    st.subheader("📊 Summary of Cross-validation Metrics")
    full_df = pd.DataFrame(full_table)
    st.table(full_df.round(3))

    # --- Radar plot ---
    st.subheader("🕸️ Radar Plot of Model Performance (interactive)")
    metrics_df = pd.DataFrame(metrics_table).set_index("Model")
    df_norm = metrics_df[["r", "R²", "Spearman", "KGE"]].fillna(0.0)
    categories = list(df_norm.columns)

    fig = go.Figure()
    for model in df_norm.index:
        values = df_norm.loc[model].tolist()
        fig.add_trace(go.Scatterpolar(
            r = values + [values[0]],
            theta = categories + [categories[0]],
            fill = 'toself',
            name = model,
            line=dict(color=color_map.get(model, "#000000")),  # use fixed color
            hovertemplate = '<b>%{text}</b><br>Metric: %{theta}<br>Score: %{r:.3f}<extra></extra>',
            text = [model]* (len(categories)+1)
        ))

    fig.update_layout(
        polar = dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, margin=dict(l=40,r=40,t=40,b=40), height=500
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Prepare simple mean values per metric ---
    # Convert to DataFrame
    df = pd.DataFrame(error_metrics_list)
    mean_df = df.groupby('Model')[['RMSE', 'MAE', 'Bias']].mean().reset_index()

    mean_df["Bias"] = mean_df["Bias"].abs()
    mean_df = mean_df.rename(columns={"Bias": "Bias (Abs)"})

    # Melt for plotting: metrics on x-axis
    plot_df = mean_df.melt(id_vars="Model", var_name="Metric", value_name="Value")

    # --- Plot simple bar chart ---
    st.subheader("📈 Mean Error Metrics per Model")
    fig = px.bar(
        plot_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        text="Value",
        color_discrete_map=color_map,  # apply fixed colors
        title="Mean Error Metrics per Model"
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title="Error Metric Value", xaxis_title="Metric")

    st.plotly_chart(fig, use_container_width=True)
