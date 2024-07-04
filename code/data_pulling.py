import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import SQLQuery


class DataLoader:
    """
    Class to load data into pandas dataframe for modelling
    """

    def __init__(self, is_sql: bool):
        """
        Constructor for DataLoader class
        :param is_sql: if data need to be loaded from sql database then True else False
        """
        self.is_sql = is_sql
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X = None
        self.y = None
        self.target_variable = None

    def pull_data_sql(self, sql_query: str) -> pd.DataFrame:
        """
        method to pull data from sql database
        :param sql_query: string containing sql query
        :return: dataframe after querying the input sql query string
        """
        query_sno = SQLQuery('snowflake')
        df = query_sno(sql_query)
        return df

    def get_data_local(self, file_path: str, is_pickle: bool = False,
                       is_csv: bool = True) -> pd.DataFrame:
        """
        method to get data from locally stored files
        :param file_path: string file path for the data to be load
        :param is_pickle: bool saying if the file is pickle
        :param is_csv: bool saying if the file is csv
        :return: dataframe after reading the local file
        """
        if not is_pickle and not is_csv:
            raise NotImplementedError("Only pickle and csv is supported.")
        if is_csv and is_pickle:
            raise Exception("File can be either csv or pickle; not both.")
        if is_csv and not is_pickle:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_pickle(file_path)
        return df

    def get_data(self, data_ref: str, extension='csv') -> pd.DataFrame:
        """
        method to get data and create dataframe irrespective of csv/pickle/sql
        :param data_ref: string, either sql query or file path
        :param extension: string if it is csv then read csv if it is pkl read pickle
        :return: dataframe after reading the data
        """
        if self.is_sql:
            return self.pull_data_sql(data_ref)
        else:
            if extension == 'csv':
                is_csv = True
                is_pickle = False
            else:
                is_csv = False
                is_pickle = True
            return self.get_data_local(data_ref, is_pickle, is_csv)

    def split_test_train(self, data: pd.DataFrame,
                         target_column: str, test_size: float = 0.30,
                         random_state: int = None):
        """
        method to create test train split
        :param data: total data
        :param target_column: target variable for model
        :param test_size: % of data need to be separated for testing
        :param random_state: random state for split
        """
        X_train, X_test, y_train, y_test = train_test_split(data.drop(target_column, axis=1),
                                                            data[target_column],
                                                            test_size=test_size,
                                                            stratify=data[target_column], random_state=random_state)
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.X = data.drop(target_column, axis=1)
        self.y = data[target_column]
        self.target_variable = target_column
