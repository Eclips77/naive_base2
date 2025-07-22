from services.trainer.data.data_loader import DataLoader
from services.trainer.data.data_cleaner import DataCleaner
from services.trainer.data.data_preprocessor import DataPreprocessor
from services.trainer.model.model_trainer import ModelTrainer
from services.trainer.model.model_saver import ModelSaver
import os

class TrainerManager:
    """
    Coordinates the full training pipeline: loading, cleaning, preprocessing, training, and saving a model.
    """
    def __init__(self, model_dir: str):
        """
        Initialize the TrainerManager.

        Args:
            model_dir (str): Directory to save the trained model.
        """
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_preprocessor = DataPreprocessor()
        self.model_trainer = ModelTrainer()
        self.model_saver = ModelSaver()
        self.model_dir = model_dir

    def run(self, data_path: str, target_column: str, model_name: str = "naive_bayes_model.pkl") -> str:
        """
        Run the full training pipeline.

        Args:
            data_path (str): Path to the data file.
            target_column (str): Name of the target column.
            model_name (str): Name for the saved model file.
        Returns:
            str: Path to the saved model file.
        """
        print("Loading data...")
        df = self.data_loader.load(data_path)
        print("Cleaning data...")
        df = self.data_cleaner.clean(df)
        print("Preprocessing data...")
        df = self.data_preprocessor.preprocess(df)
        print("Training model...")
        model = self.model_trainer.train(df, target_column)
        print("Saving model...")
        model_path = self.model_saver.save(model, self.model_dir, model_name)
        print(f"Model saved to {model_path}")
        return model_path 