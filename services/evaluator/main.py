import argparse
import os
from services.evaluator.evaluator_manager import EvaluatorManager

def main():
    """
    Entry point for the Evaluator service. Parses arguments and runs the evaluation pipeline using EvaluatorManager.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a test dataset.")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the trained model file (pickle)")
    parser.add_argument("--test-file", type=str, required=True, help="Path to the test dataset file (CSV, Excel, or JSON)")
    parser.add_argument("--target-column", type=str, required=True, help="Name of the target column")
    args = parser.parse_args()

    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models")
    models_dir = os.path.abspath(models_dir)
    manager = EvaluatorManager(models_dir=models_dir)
    manager.run(model_name=args.model_name, test_file=args.test_file, target_column=args.target_column)

if __name__ == "__main__":
    main() 