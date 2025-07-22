import pandas as pd
import os

class DataLoader:
    """
    Loads data from a file (CSV, Excel, or JSON) and returns a pandas DataFrame.
    """
    @staticmethod
    def load(file_path: str) -> pd.DataFrame:
        """
        Load data from the specified file path.

        Args:
            file_path (str): Path to the data file.
        Returns:
            pd.DataFrame: Loaded data.
        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is not supported.
        """
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File '{file_path}' not found.")
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".csv":
            return pd.read_csv(file_path)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        elif ext == ".json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Supported formats: CSV, Excel, JSON.") 