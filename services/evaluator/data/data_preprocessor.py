import pandas as pd

class DataPreprocessor:
    """
    Preprocesses a pandas DataFrame (e.g., encodes categorical variables).
    """
    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to preprocess.
        Returns:
            pd.DataFrame: Preprocessed DataFrame.
        """
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        return df 