# Trainer Service

This service is responsible for training a Naive Bayes model on a given dataset and saving the trained model to a file (in a shared volume).

## Usage

- Trains a model on a specified data file.
- Saves the trained model file to `/models` (shared volume).

## Run

```bash
python main.py --data-file <file.csv|file.xlsx|file.json> --target-column <target> [--model-name <model_name.pkl>]
```

## Input
- Data file (CSV, Excel, or JSON) from any path.
- Target column name.

## Output
- Trained model file (pickle) in the `/models` directory. 