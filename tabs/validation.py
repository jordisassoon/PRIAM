import streamlit as st
import pandas as pd
import numpy as np
from utils.dataloader import ProxyDataLoader
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WAPLS
from models.rf import RF
from validation.cross_validate import run_grouped_cv
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error
import io

# Define a fixed color map for models
color_map = {
    "MAT": "#1f77b4",  # blue
    "BRT": "#ff7f0e",  # orange
    "RF": "#2ca02c",  # green
}


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
):

    st.header("Model Validation")
    st.info(
        "Evaluate model performance using grouped cross-validation and diagnostic metrics (RMSE, MAE, R², r, KGE, Bias)."
    )

    # --- Check inputs ---
    if not (train_climate_file and train_proxy_file):
        st.warning("Please upload both climate and proxy training datasets.")
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

    # --- Load data ---
    loader = ProxyDataLoader(
        climate_file=train_climate_file,
        proxy_file=train_proxy_file,
        test_file=test_proxy_file,
        mask_file=taxa_mask_file,
    )

    X_train, y_train, obs_names = loader.load_training_data(target)

    # --- Available models ---
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
        "RF": (
            RF,
            {"n_estimators": rf_trees, "max_depth": 6, "random_state": random_seed},
        ),
    }

    if model_choice == "All":
        models_to_run = available_models
    else:
        models_to_run = {model_choice: available_models[model_choice]}

    metrics_table = []
    full_table = []
    error_metrics_list = []

    st.subheader(f"Running Cross-validation...")
    # --- Run CV for each model ---
    for name, (model_class, params) in models_to_run.items():

        with st.spinner(f"Running {cv_folds}-fold grouped CV on {name}..."):
            scores = run_grouped_cv(
                model_class,
                params,
                X_train,
                y_train,
                obs_names,
                n_splits=cv_folds,
                seed=random_seed,
                loader=loader,
            )

        # Store means (numeric) for plotting
        metrics_table.append(
            {
                "Model": name,
                "R²": np.mean(scores["r2"]),
                "Pearson R": np.mean(scores["r"]),
                "Spearman": np.mean(scores["spearman"]),
                "KGE": np.mean(scores["kge"]),
            }
        )

        # Store mean ± std
        full_table.append(
            {
                "Model": name,
                "RMSE": f"{np.mean(scores['rmse']):.2f} ± {np.std(scores['rmse']):.2f}",
                "MAE": f"{np.mean(scores['mae']):.2f} ± {np.std(scores['mae']):.2f}",
                "R²": f"{np.mean(scores['r2']):.2f} ± {np.std(scores['r2']):.2f}",
                "Pearson R": f"{np.mean(scores['r']):.2f} ± {np.std(scores['r']):.2f}",
                "Spearman": f"{np.mean(scores.get('spearman', [0])):.2f} ± {np.std(scores.get('spearman', [0])):.2f}",
                "KGE": f"{np.mean(scores['kge']):.2f} ± {np.std(scores['kge']):.2f}",
                "Bias": f"{np.mean(scores['bias']):.2f} ± {np.std(scores['bias']):.2f}",
            }
        )

        # Store errors for histogram
        for rmse, mae, bias in zip(scores["rmse"], scores["mae"], scores["bias"]):
            error_metrics_list.append(
                {"Model": name, "RMSE": rmse, "MAE": mae, "Bias": bias}
            )

    # --- Display full metrics table ---
    st.subheader("Summary of Cross-validation Metrics")
    full_df = pd.DataFrame(full_table).round(3)

    # --- Display table ---
    st.dataframe(full_df)

    # --- Radar plot ---
    st.subheader("Radar Plot of Model Performance")
    metrics_df = pd.DataFrame(metrics_table).set_index("Model")
    df_norm = metrics_df[["Pearson R", "R²", "Spearman", "KGE"]].fillna(0.0)
    categories = list(df_norm.columns)

    # df_norm["R²"] = df_norm["R²"] + 1.0 / 2  # shift R² to [0, 1] for better visibility
    # df_norm["KGE"] = (df_norm["KGE"] + 1.0) / 2  # shift KGE to [0, 1]
    # df_norm["Pearson R"] = (df_norm["Pearson R"] + 1.0) / 2  # shift R to [0, 1]

    fig = go.Figure()
    for model in df_norm.index:
        values = df_norm.loc[model].tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=categories + [categories[0]],
                fill="toself",
                name=model,
                line=dict(color=color_map.get(model, "#000000")),  # use fixed color
                hovertemplate="<b>%{text}</b><br>Metric: %{theta}<br>Score: %{r:.3f}<extra></extra>",
                text=[model] * (len(categories) + 1),
            )
        )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        margin=dict(l=40, r=40, t=40, b=40),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.info("Note: R², KGE, and Pearson R values are shifted for visualization purposes.")

    # --- Prepare simple mean values per metric ---
    # Convert to DataFrame
    df = pd.DataFrame(error_metrics_list)
    mean_df = df.groupby("Model")[["RMSE", "MAE", "Bias"]].mean().reset_index()

    mean_df["Bias"] = mean_df["Bias"].abs()
    mean_df = mean_df.rename(columns={"Bias": "Bias (Abs)"})

    # Melt for plotting: metrics on x-axis
    plot_df = mean_df.melt(id_vars="Model", var_name="Metric", value_name="Value")

    # --- Plot simple bar chart ---
    st.subheader("Mean Error Metrics per Model")
    fig = px.bar(
        plot_df,
        x="Metric",
        y="Value",
        color="Model",
        barmode="group",
        text="Value",
        color_discrete_map=color_map,  # apply fixed colors
        title="Mean Error Metrics per Model",
    )

    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(yaxis_title="Error Metric Value", xaxis_title="Metric")

    st.plotly_chart(fig, use_container_width=True)

    # --- Scree Plot Toggle ---
    show_scree = st.toggle("Show Scree Plot for Model Sensitivity", value=False)

    if show_scree:
        st.info(
            "This analysis shows how model performance changes with key hyperparameters "
            "(neighbors for MAT, trees for BRT/RF)."
        )

        # Define parameter ranges for each model
        param_ranges = {
            "MAT": {"param_name": "n_neighbors", "values": [1, 2, 3, 4, 5, 6, 7]},
            "RF": {"param_name": "n_estimators", "values": [50, 100, 200, 300, 500, 700, 1000]},
            "BRT": {"param_name": "n_estimators", "values": [50, 100, 200, 300, 500, 700, 1000]},
        }

        for name, (model_class, base_params) in models_to_run.items():
            if name not in param_ranges:
                continue  # skip models without scree logic

            st.markdown(f"### {name} Scree Plot")

            param_name = param_ranges[name]["param_name"]
            test_values = param_ranges[name]["values"]

            scree_results = []

            for val in test_values:
                # Update model params for this run
                params = base_params.copy()
                params[param_name] = val

                with st.spinner(f"Running {cv_folds}-fold CV with {param_name}={val}..."):
                    scores = run_grouped_cv(
                        model_class,
                        params,
                        X_train,
                        y_train,
                        obs_names,
                        n_splits=cv_folds,
                        seed=random_seed,
                        loader=loader,
                    )

                scree_results.append(
                    {
                        param_name: val,
                        "R²": np.mean(scores["r2"]),
                        "RMSE": np.mean(scores["rmse"]),
                        "MAE": np.mean(scores["mae"]),
                    }
                )

            scree_df = pd.DataFrame(scree_results)

            # --- Plot Scree Plot (R² vs parameter) ---
            fig = go.Figure()

            for metric in ["R²", "RMSE", "MAE"]:
                fig.add_trace(
                    go.Scatter(
                        x=scree_df[param_name],
                        y=scree_df[metric],
                        mode="lines+markers",
                        name=metric,
                    )
                )

            fig.update_layout(
                title=f"Scree Plot for {name} ({param_name})",
                xaxis_title=param_name,
                yaxis_title="Metric Value",
                legend_title="Metric",
                height=500,
            )

            st.plotly_chart(fig, use_container_width=True)
