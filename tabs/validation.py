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
    X_train,
    y_train,
    loader,
    train_metadata,
):

    st.header("Model Validation")

    if X_train is None:
        st.warning("Please upload training datasets in the 'Data Loading' section.")
        return

    # --- Available models ---
    available_models = {
        "MAT": (MAT, {"n_neighbors": st.session_state.get("n_neighbors", None)}),
        "BRT": (
            BRT,
            {
                "n_estimators": st.session_state.get("brt_trees", None),
                "learning_rate": st.session_state.get("brt_learning_rate", None),
                "max_depth": st.session_state.get("brt_max_depth", None),
                "random_state": st.session_state.get("random_seed", None),
            },
        ),
        "RF": (
            RF,
            {
                "n_estimators": st.session_state.get("rf_trees", None),
                "max_depth": st.session_state.get("rf_max_depth", None),
                "random_state": st.session_state.get("random_seed", None),
            },
        ),
    }

    metrics_table = []
    full_table = []
    error_metrics_list = []

    # --- Run CV for each model ---
    for name, (model_class, params) in available_models.items():

        with st.spinner(f"Running {st.session_state['cv_folds']}-fold grouped CV on {name}..."):
            scores = run_grouped_cv(
                model_class,
                params,
                X_train,
                y_train,
                train_metadata["OBSNAME"],
                n_splits=st.session_state["cv_folds"],
                seed=st.session_state["random_seed"],
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
            error_metrics_list.append({"Model": name, "RMSE": rmse, "MAE": mae, "Bias": bias})

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
            "MAT": {
                "param_name": "n_neighbors",
                "values": [1, 2, 3, 4, 5, 6, 7],
            },
            "RF": {
                "param_name": "n_estimators",
                "values": [1, 10, 50, 100, 200, 300, 500, 700, 1000],
            },
            "BRT": {
                "param_name": "n_estimators",
                "values": [1, 10, 50, 100, 200, 300, 500, 700, 1000],
            },
        }

        for name, (model_class, base_params) in available_models.items():
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

                with st.spinner(f"Running {st.session_state['cv_folds']}-fold CV with {param_name}={val}..."):
                    scores = run_grouped_cv(
                        model_class,
                        params,
                        X_train,
                        y_train,
                        train_metadata["OBSNAME"],
                        n_splits=st.session_state["cv_folds"],
                        seed=st.session_state["random_seed"],
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
