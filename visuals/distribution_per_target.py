import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import click

@click.command()
@click.argument("pollen_file", type=click.Path(exists=True))
@click.argument("climate_file", type=click.Path(exists=True))
@click.option("--id-col", default="OBSNAME", help="Column name to merge datasets on")
@click.option("--target-col", required=True, help="Climate variable column to plot against")
@click.option("--taxa", required=True, help="Taxa column to plot")
@click.option("--bins", default=5, help="Number of bins for target variable")
@click.option("--output-file", default="pollen_preference.png", help="Output image file")
def main(pollen_file, climate_file, id_col, target_col, taxa, bins, output_file):
    """Plot a bar chart of taxa preference per binned target climate variable."""

    # ==== LOAD DATA ====
    pollen_df = pd.read_csv(pollen_file, delimiter=',', encoding="latin1")
    climate_df = pd.read_csv(climate_file, delimiter=',', encoding="latin1")

    # ==== MERGE DATASETS ====
    merged_df = pd.merge(pollen_df, climate_df, on=id_col, how='inner')

    if taxa not in pollen_df.columns:
        raise ValueError(f"Taxa '{taxa}' not found in pollen dataset")

    # ==== BIN TARGET VARIABLE ====
    merged_df['binned_target'] = pd.cut(merged_df[target_col], bins=bins)

    # ==== CALCULATE TAXA PREFERENCE ====
    # Sum pollen counts per bin for the taxa
    taxa_sum = merged_df.groupby('binned_target')[taxa].sum()
    # Sum pollen counts per bin for all samples to normalize
    total_sum = merged_df.groupby('binned_target')[taxa].count()
    preference = (taxa_sum / total_sum).reset_index()

    # ==== PLOT BAR CHART ==== 
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='binned_target', y=taxa, data=preference)
    plt.ylabel(f'{taxa} Average Count')
    plt.xlabel(target_col)
    
    # ==== SET X-AXIS LABELS TO BIN MID POINTS ====
    ax.set_xticklabels([f"{interval.mid:.2f}" for interval in preference['binned_target']], rotation=45, ha='right')
    plt.tight_layout()

    # ==== SAVE PLOT ====
    plt.savefig(output_file, dpi=300)
    print(f"Pollen preference bar chart for {taxa} saved to {output_file}")

if __name__ == "__main__":
    main()