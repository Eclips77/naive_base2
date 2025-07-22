import pandas as pd

class DataCleaner:
    """
    Cleans a pandas DataFrame (e.g., removes duplicates, handles missing values).
    """
    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the given DataFrame.

        Args:
            df (pd.DataFrame): The DataFrame to clean.
        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df.drop_duplicates()
        df = df.dropna()
        return df 