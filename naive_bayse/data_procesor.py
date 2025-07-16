from sklearn.model_selection import train_test_split
import pandas as pd

class DataProcessor:
    """Utility class for preprocessing datasets."""

    def __init__(self, df):
        """Store a DataFrame for later processing.

        Args:
            df (pandas.DataFrame): Raw data.
        """
        self.df = df

    def clean_data(self):
        """Remove rows with missing values.

        Returns:
            pandas.DataFrame: Cleaned DataFrame.

        Usage:
            cleaned = processor.clean_data()
        """
        self.df.dropna(inplace=True)
        return self.df

    # def get_features(self) -> pd.DataFrame:
    #     """
    #         Get the features from the DataFrame.
    
    #         Returns:
    #             pd.DataFrame: DataFrame containing the features, or an empty DataFrame if no data is loaded.
    #         """
    #     if self.df is not None:
    #         if self.__label_column in self.df.columns:
    #             # Drop the label column to return only features
    #             return self.df.drop(columns=[self.__label_column], errors='ignore')
    #     return pd.DataFrame()

    # def get_labels(self) -> pd.Series:
    #     """
    #         Get the labels from the DataFrame.
    #
    #         Returns:
    #             pd.Series: Series containing the labels, or an empty Series if no data is loaded.
    #         """
    #     if self.__data is not None and self.__label_column in self.__data.columns:
    #         return (
    #             self.__data[self.__label_column]
    #             .astype(str)
    #             .str.strip()
    #             .str.lower()
    #         )
    #
    #     return pd.Series(dtype=float)

    def split_data(self, train_size=0.7, random_state=42):
        """Split the dataset into train and test parts.

        Args:
            train_size (float): Fraction of data to use for training.
            random_state (int): Random seed for reproducibility.

        Returns:
            Tuple[pandas.DataFrame, pandas.DataFrame]: Train and test sets.

        Usage:
            train_df, test_df = processor.split_data()
        """
        train_df, test_df = train_test_split(
            self.df, train_size=train_size, random_state=random_state
        )
        return train_df, test_df