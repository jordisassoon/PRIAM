import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d

@click.command()
@click.option('--predictions_csv', required=True, type=click.Path(exists=True), help='CSV file containing predictions')
@click.option('--depth_csv', required=True, type=click.Path(exists=True), help='CSV file containing depth values')
@click.option('--output_file', required=True, type=click.Path(), help='Path to save the visualization image')
@click.option('--title', default='Pollen-based Reconstruction', help='Title of the plot')
@click.option('--smooth_sigma', default=2.0, type=float, help='Sigma for Gaussian smoothing')
def main(predictions_csv, depth_csv, output_file, title, smooth_sigma):
    # Load predictions
    df_pred = pd.read_csv(predictions_csv)
    predictions = df_pred.iloc[:, 1].values  # assume second column contains predictions

    # Load depth values
    df_depth = pd.read_csv(depth_csv)
    if 'Age' not in df_depth.columns:
        raise ValueError("Age CSV must contain a 'age' column")
    depth = df_depth['Age'].values

    if len(depth) != len(predictions):
        raise ValueError(f"Number of depth values ({len(depth)}) does not match number of predictions ({len(predictions)})")

    # Smooth predictions
    predictions_smooth = gaussian_filter1d(predictions, sigma=smooth_sigma)

    plt.figure(figsize=(10, 5))

    # Thin jagged line (no markers)
    plt.plot(depth, predictions, linestyle='-', color='tab:blue', linewidth=1, alpha=0.5)

    # Thick smoothed line
    plt.plot(depth, predictions_smooth, linestyle='-', color='tab:red', linewidth=3)

    plt.title(title)
    plt.xlabel('Age')
    plt.ylabel('Predicted environmental value')
    plt.grid(True)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Visualization saved to {output_file}')

if __name__ == '__main__':
    main()
