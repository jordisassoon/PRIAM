import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

@click.command()
@click.option('--predictions_csv', required=True, type=click.Path(exists=True), help='CSV file containing predictions')
@click.option('--output_file', required=True, type=click.Path(), help='Path to save the visualization image')
@click.option('--title', default='Pollen-based Reconstruction', help='Title of the plot')
def main(predictions_csv, output_file, title):
    # Load predictions from CSV
    df = pd.read_csv(predictions_csv)
    predictions = df.iloc[:, 0].values  # assume first column contains predictions

    # Prepare sequential time axis
    n_samples = len(predictions)
    time_points = np.arange(n_samples)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(time_points, predictions, marker='o', linestyle='-', color='tab:blue')
    plt.title(title)
    plt.xlabel('Sample / Time point')
    plt.ylabel('Predicted environmental value')
    plt.grid(True)

    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f'Visualization saved to {output_file}')

if __name__ == '__main__':
    main()
