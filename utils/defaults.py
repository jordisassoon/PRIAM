def get_default_state_config():
    # --- Initialize session state defaults ---
    return {
        "model_choice": "MAT",
        "model_choices": ["MAT", "BRT", "RF", "All"],
        "target": "TANN",
        "target_cols": ["TANN", "PANN", "MTWA", "MTCO"],
        "taxa_cols": {},
        "n_neighbors": 5,
        "brt_trees": 200,
        "rf_trees": 200,
        "cv_folds": 3,
        "random_seed": 42,
        "prediction_axis": "Age",
        "initialized": False,
        "uploaded_state": None,
        "train_climate_file": None,
        "train_proxy_file": None,
        "test_proxy_file": None,
        "taxa_mask_file": None,
        "coords_file": None,
        "use_dummy": False,
    }
