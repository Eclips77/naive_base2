from sklearn.model_selection import train_test_split
import pandas as pd

class DataProcessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
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
        train_df, test_df = train_test_split(self.df, train_size=train_size, random_state=random_state)
        return train_df, test_df