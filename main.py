import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.dataloader import PollenDataLoader
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WA_PLS
from models.rf import RF

@click.command()
@click.option('--train_climate', required=True, type=click.Path(exists=True), help='Path to climate target CSV file')
@click.option('--train_pollen', required=True, type=click.Path(exists=True), help='Path to pollen training CSV file')
@click.option('--test_pollen', required=True, type=click.Path(exists=True), help='Path to test fossil pollen CSV file')
@click.option('--model', required=True, type=click.Choice(['MAT', 'BRT', 'WA-PLS', 'RF'], case_sensitive=False), help='Which model to use')
@click.option('--target', default='TANN', help='Climate target variable to predict (e.g., TANN)')
@click.option('--k', default=3, type=int, help='Number of neighbors for MAT')
@click.option('--seed', default=42, type=int, help='Random seed')
@click.option('--pls_components', default=3, type=int, help='Number of PLS components for WA-PLS')
@click.option('--output_csv', required=True, type=click.Path(), help='Path to save predictions CSV')
def main(train_climate, train_pollen, test_pollen, model, target, k, seed, pls_components, output_csv):
    np.random.seed(seed)

    # Load and prepare data
    loader = PollenDataLoader(
        climate_file=train_climate,
        pollen_file=train_pollen,
        test_file=test_pollen
    )

    X_train, y_train = loader.load_training_data(target=target)
    X_test, ages = loader.load_test_data()
    X_train_aligned, X_test_aligned = loader.align_taxa(X_train, X_test)

    # Train and predict
    if model.upper() == 'MAT':
        mat_model = MAT(k=k)
        mat_model.fit(X_train_aligned, y_train)
        predictions = mat_model.predict_with_progress(X_test_aligned, batch_size=50)

    elif model.upper() == 'BRT':
        brt_model = BRT(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=seed)
        brt_model.fit(X_train_aligned, y_train)
        predictions = brt_model.predict_with_progress(X_test_aligned, batch_size=50)

    elif model.upper() == 'WA-PLS':
        wa_pls_model = WA_PLS(n_components=pls_components)
        wa_pls_model.fit(X_train_aligned, y_train)
        predictions = wa_pls_model.predict_with_progress(X_test_aligned, batch_size=50)

    elif model.upper() == 'RF':
        rf_model = RF(n_estimators=200, max_depth=10, random_state=42)
        rf_model.fit(X_train_aligned, y_train)
        predictions = rf_model.predict_with_progress(X_test_aligned, batch_size=50)

    else:
        print(f"Your model {model} was not found")
        return

    # Save predictions to CSV for visualization
    pd.DataFrame(predictions, columns=[f'Predicted_{target}']).to_csv(output_csv, index=False)
    print(f"Predictions for {target} using {model} model saved to {output_csv}")

if __name__ == '__main__':
    main()