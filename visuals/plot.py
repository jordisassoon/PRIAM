import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


@click.command()
@click.option(
    "--predictions_csv",
    required=True,
    type=click.Path(exists=True),
    help="CSV file containing predictions",
)
@click.option(
    "--depth_csv",
    required=True,
    type=click.Path(exists=True),
    help="CSV file containing depth values",
)
@click.option(
    "--output_file",
    required=True,
    type=click.Path(),
    help="Path to save the visualization image",
)
@click.option("--title", default="Pollen-based Reconstruction", help="Title of the plot")
@click.option(
    "--smooth_sigma",
    default=2.0,
    type=float,
    help="Sigma for Gaussian smoothing",
)
def main(predictions_csv, depth_csv, output_file, title, smooth_sigma):
    # Load predictions
    df_pred = pd.read_csv(predictions_csv)
    prediction_cols = df_pred.columns[1:]  # assume first column is not a prediction
    predictions = df_pred[prediction_cols].values  # shape: (n_rows, n_columns)

    # Load depth values
    df_depth = pd.read_csv(depth_csv)
    if "Age" not in df_depth.columns:
        raise ValueError("Age CSV must contain an 'Age' column")
    depth = df_depth["Age"].values

    if len(depth) != predictions.shape[0]:
        raise ValueError(
            f"Number of depth values ({len(depth)}) does not match number of predictions ({predictions.shape[0]})"
        )

    plt.figure(figsize=(10, 5))

    # Loop through each prediction column
    for i, col_name in enumerate(prediction_cols):
        pred_col = predictions[:, i]

        # Thin jagged line
        plt.plot(
            depth,
            pred_col,
            linestyle="-",
            color="tab:blue",
            linewidth=1,
            alpha=0.5,
        )

        # Smoothed line (apply Gaussian filter column-wise)
        pred_smooth = gaussian_filter1d(pred_col, sigma=smooth_sigma)
        plt.plot(depth, pred_smooth, linestyle="-", color="tab:red", linewidth=3)

    plt.title(title)
    plt.xlabel("Age")
    plt.ylabel("Predicted environmental value")
    plt.grid(True)

    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Visualization saved to {output_file}")


if __name__ == "__main__":
    main()
