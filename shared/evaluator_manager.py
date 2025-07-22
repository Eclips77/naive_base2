from shared.model_loader import ModelLoader
from shared.data_loader import DataLoader
from shared.data_cleaner import DataCleaner
from shared.data_preprocessor import DataPreprocessor
from shared.model_evaluator import ModelEvaluator
import os

class EvaluatorManager:
    """
    Coordinates the full evaluation pipeline: loading model, loading/cleaning/preprocessing test data, and evaluating.
    """
    def __init__(self, models_dir: str):
        """
        Initialize the EvaluatorManager.

        Args:
            models_dir (str): Directory containing trained model files.
        """
        self.models_dir = models_dir
        self.model_loader = ModelLoader()
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_preprocessor = DataPreprocessor()
        self.model_evaluator = ModelEvaluator()

    def run(self, model_name: str, test_file: str, target_column: str) -> float:
        """
        Run the full evaluation pipeline.

        Args:
            model_name (str): Name of the trained model file.
            test_file (str): Path to the test dataset file.
            target_column (str): Name of the target column.
        Returns:
            float: Accuracy score.
        """
        print("Loading model...")
        model = self.model_loader.load(self.models_dir, model_name)
        print("Loading test data...")
        df = self.data_loader.load(test_file)
        print("Cleaning test data...")
        df = self.data_cleaner.clean(df)
        print("Preprocessing test data...")
        df = self.data_preprocessor.preprocess(df)
        print("Evaluating model...")
        accuracy = self.model_evaluator.evaluate(model, df, target_column)
        print(f"Accuracy: {accuracy:.4f}")
        return accuracy 