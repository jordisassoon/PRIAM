import click
import pandas as pd
import numpy as np

from utils.dataloader import PollenDataLoader
from utils.cross_validate import run_grouped_cv
from models.mat import MAT
from models.brt import BRT
from models.wa_pls import WA_PLS
from models.rf import RF

@click.command()
@click.option('--train_climate', required=True, type=click.Path(exists=True))
@click.option('--train_pollen', required=True, type=click.Path(exists=True))
@click.option('--test_pollen', required=True, type=click.Path(exists=True))
@click.option('--taxa_mask', default=None, type=click.Path(exists=True))
@click.option('--model_name', required=True,
              type=click.Choice(['MAT', 'BRT', 'WA-PLS', 'RF', 'ALL'], case_sensitive=False))
@click.option('--target', default='TANN', show_default=True)
@click.option('--seed', default=42, type=int, show_default=True)
@click.option('--cv_folds', default=1, type=int, show_default=True, help='Number of CV folds (grouped by OBSNAME)')

# MAT options
@click.option('--k', default=3, type=int, show_default=True)

# BRT options
@click.option('--brt_estimators', default=200, type=int, show_default=True)
@click.option('--brt_lr', default=0.05, type=float, show_default=True)
@click.option('--brt_max_depth', default=4, type=int, show_default=True)

# WA-PLS options
@click.option('--pls_components', default=3, type=int, show_default=True)

# RF options
@click.option('--rf_estimators', default=200, type=int, show_default=True)
@click.option('--rf_max_depth', default=10, type=int, show_default=True)

@click.option('--output_csv', required=True, type=click.Path())
def main(train_climate, train_pollen, test_pollen, taxa_mask, model_name, target, seed, cv_folds,
         k, brt_estimators, brt_lr, brt_max_depth,
         pls_components, rf_estimators, rf_max_depth, output_csv):

    np.random.seed(seed)

    # --- Load data ---
    loader = PollenDataLoader(train_climate, train_pollen, test_pollen, taxa_mask)
    X_train, y_train, obs_names = loader.load_training_data(target=target)
    X_test, ages = loader.load_test_data()
    X_train_aligned, X_test_aligned = loader.align_taxa(X_train, X_test)

    # --- Define models ---
    model_configs = {
        "MAT": (MAT, {"k": k}),
        "BRT": (BRT, {"n_estimators": brt_estimators,
                      "learning_rate": brt_lr,
                      "max_depth": brt_max_depth,
                      "random_state": seed}),
        "WA-PLS": (WA_PLS, {"n_components": pls_components}),
        "RF": (RF, {"n_estimators": rf_estimators,
                    "max_depth": rf_max_depth,
                    "random_state": seed})
    }

    # Select models
    if model_name.upper() == "ALL":
        models_to_run = model_configs
    else:
        models_to_run = {model_name.upper(): model_configs[model_name.upper()]}

    predictions_dict = {}

    for name, (model_class, params) in models_to_run.items():
        print(f"\n=== Running {model_class} ===")

        # --- Cross-validation ---
        if cv_folds > 1:
            run_grouped_cv(model_class, params, X_train_aligned, y_train, obs_names,
                           n_splits=cv_folds, seed=seed, loader=loader)

        # --- Final fit + predict ---
        model = model_class(**params)
        model.fit(X_train_aligned, y_train)
        preds = model.predict(X_test_aligned)
        predictions_dict[model_class] = preds

    # --- Save predictions ---
    df_out = pd.DataFrame({"Age": ages})
    for name, preds in predictions_dict.items():
        df_out[f"{name}_Predicted_{target}"] = preds

    df_out.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to {output_csv}")


if __name__ == '__main__':
    main()
