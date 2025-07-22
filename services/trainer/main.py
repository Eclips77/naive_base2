import argparse
import os
from services.trainer.trainer_manager import TrainerManager

def main():
    """
    Entry point for the Trainer service. Parses arguments and runs the training pipeline using TrainerManager.
    """
    parser = argparse.ArgumentParser(description="Train a Naive Bayes model on a dataset and save the model to a file.")
    parser.add_argument("--data-file", type=str, required=True, help="Path to the data file (CSV, Excel, or JSON)")
    parser.add_argument("--target-column", type=str, required=True, help="Name of the target column")
    parser.add_argument("--model-name", type=str, default="naive_bayes_model.pkl", help="Name for the saved model file")
    args = parser.parse_args()

    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models")
    model_dir = os.path.abspath(model_dir)
    manager = TrainerManager(model_dir=model_dir)
    manager.run(data_path=args.data_file, target_column=args.target_column, model_name=args.model_name)

if __name__ == "__main__":
    main() 