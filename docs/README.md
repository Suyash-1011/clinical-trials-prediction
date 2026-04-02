# Clinical Trials Outcome Prediction - Documentation

## Project Description
A modular, production-grade ML pipeline for predicting clinical trial outcomes. Includes robust data processing, feature engineering, model training, evaluation, reporting, and CLI automation.

## Modules
- **src/data.py**: Data loading and cleaning
- **src/features.py**: Feature engineering and selection
- **src/models.py**: Model training, prediction, and persistence
- **src/evaluate.py**: Evaluation and reporting utilities
- **src/utils.py**: General utilities (reproducibility, logging)
- **src/plotting.py**: Plotting and image integration

## CLI Scripts
- **scripts/data_cli.py**: Data cleaning
- **scripts/train_cli.py**: Model training
- **scripts/eval_cli.py**: Model evaluation
- **scripts/predict_cli.py**: Prediction
- **scripts/pipeline_cli.py**: End-to-end pipeline

## Testing
Run all tests:
```sh
pytest tests/
```

## Extending
- Add new models in src/models.py
- Add new feature engineering in src/features.py
- Add new CLI tools in scripts/

## License
MIT License
