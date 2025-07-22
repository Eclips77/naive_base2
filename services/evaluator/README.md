# Evaluator Service

This service evaluates the performance of a trained model on a test dataset.

## Usage

- Loads a trained model (pickle) from a shared volume (`/models`).
- Loads a test dataset.
- Computes accuracy and other metrics.

## Run

```bash
python main.py --test-file <file.csv|file.xlsx|file.json> --target-column <target> [--model-name <model_name.pkl>]
```

## Input
- Trained model file (pickle) in the `/models` directory.
- Test dataset file (CSV, Excel, or JSON).
- Target column name.

## Output
- Evaluation results (accuracy, etc.) to standard output. 