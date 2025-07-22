import os
from naive_bayes.data_loader import DataLoader
from src.naive_bayes_classifier import NaiveBayesClassifier
from naive_bayes.data_processor import DataProcessor
from src.model_evaluator import Evaluator


class App:
    def __init__(self):
        """Initialize the App with no loaded data or target column."""
        self.classifier = NaiveBayesClassifier()
        self.train_df = None
        self.test_df = None
        self.target_column = None

    def list_available_files(self):
        """List all CSV files available in the Data directory.

        Returns:
            List[str]: A list of CSV file names.

        Usage:
            files = app.list_available_files()
        """
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "Data")
        return [f for f in os.listdir(data_dir) if f.endswith('.csv')]

    def load_and_clean(self, selected_file_name):
        """Load the selected CSV file, clean it and split into train/test.

        Args:
            selected_file_name (str): Name of the CSV file to load.

        Returns:
            dict: Status message and list of column names.

        Usage:
            app.load_and_clean("data.csv")
        """
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(BASE_DIR, "Data")
        selected_file = os.path.join(data_dir, selected_file_name)

        if not os.path.isfile(selected_file):
            raise FileNotFoundError(f"File '{selected_file_name}' not found in Data folder.")

        df = DataLoader.load_data(selected_file)
        processor = DataProcessor(df)
        processor.clean_data()
        self.train_df, self.test_df = processor.split_data()
        return {"status": "success", "columns": self.train_df.columns.tolist()}

    def set_target_column(self, column_name):
        """Set the target column to be used as labels for training and prediction.

        Args:
            column_name (str): Name of the column to use as target.

        Returns:
            dict: Status message with the selected target column.

        Usage:
            app.set_target_column("label")
        """
        if self.train_df is None:
            raise ValueError("No data loaded yet.")
        if column_name not in self.train_df.columns:
            raise ValueError(f"Column '{column_name}' does not exist.")
        self.target_column = column_name
        return {"status": "success", "target_column": self.target_column}

    def get_features_with_values(self):
        """Get all feature columns (excluding the target) with their unique possible values.

        Returns:
            dict: Mapping from feature name to list of unique values.

        Usage:
            features = app.get_features_with_values()
        """
        if self.train_df is None:
            raise ValueError("No training data loaded.")
        if self.target_column is None:
            raise ValueError("Target column not set.")

        features = {}
        for col in self.train_df.columns:
            if col != self.target_column:
                features[col] = sorted(self.train_df[col].unique().tolist())
        return features

    def train_model(self):
        """Train the Naive Bayes classifier on the loaded training data.

        Returns:
            dict: Status message.

        Usage:
            app.train_model()
        """
        if self.train_df is None:
            raise ValueError("No training data loaded.")
        if self.target_column is None:
            raise ValueError("Target column not set.")
        self.classifier.fit(self.train_df, self.target_column)
        return {"status": "success", "message": "Model trained successfully."}

    def evaluate_model(self):
        """Evaluate the trained model using the test dataset.

        Returns:
            dict: Accuracy score.

        Usage:
            app.evaluate_model()
        """
        if self.test_df is None:
            raise ValueError("No test data loaded.")
        if self.target_column is None:
            raise ValueError("Target column not set.")
        evaluator = Evaluator()
        accuracy = evaluator.evaluate(self.classifier, self.test_df, self.target_column)
        return {"accuracy": accuracy}

    def classify_record(self, record: dict):
        """Classify a new record using the trained model.

        Args:
            record (dict): Dictionary of feature values.

        Returns:
            dict: Prediction result.

        Usage:
            result = app.classify_record(record)
        """
        if self.train_df is None:
            raise ValueError("No data loaded.")
        if self.target_column is None:
            raise ValueError("Target column not set.")
        prediction = self.classifier.classify(record)
        return {"prediction": prediction}
