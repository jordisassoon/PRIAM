import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click

@click.command()
@click.argument("climate_file", type=click.Path(exists=True))
@click.option("--var", required=True, help="Climate variable to plot")
@click.option("--output-file", default="climate_histogram.png", help="Output image file")
def main(climate_file, var, output_file):
    """Plot a histogram of a single climate variable."""

    # ==== LOAD DATA ====
    climate_df = pd.read_csv(climate_file, delimiter=',', encoding='latin1')

    if var not in climate_df.columns:
        raise ValueError(f"Column '{var}' not found in climate dataset")

    # ==== PLOT HISTOGRAM ====
    plt.figure(figsize=(8, 6))
    sns.histplot(climate_df[var], bins=20, kde=True)
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {var}')
    plt.tight_layout()

    # ==== SAVE PLOT ====
    plt.savefig(output_file, dpi=300)
    print(f"Histogram of {var} saved to {output_file}")

if __name__ == "__main__":
    main()