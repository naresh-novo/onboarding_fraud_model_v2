import numpy as np
import pandas as pd


class EDA:
    """
    class for doing basic EDA
    """

    def __init__(self, X_train, X_test, y_train, y_test, target_variable):
        """
        constructor for EDA
        :param X_train: Independent variables of training data
        :param X_test: Independent variables of testing data
        :param y_train: target variable for train
        :param y_test: target variable for test
        :param target_variable: target variable for model
        """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.target_variable = target_variable
        self.variance_df = pd.DataFrame([])
        self.extreme_values_df = pd.DataFrame([])
        self.missing_values_df = pd.DataFrame([])
        self.cardinality_df = pd.DataFrame([])
        self.numeric_numeric_corr_df = pd.DataFrame([])
        self.vif_df = pd.DataFrame([])

    def variance(self, is_removal=True, threshold=.1):
        """
        Variance for each numeric columns in train df
        :param is_removal: if columns need to be dropped based on variance
        :param threshold: threshold cutoff for variance
        """
        self.variance_df = self.X_train.var().sort_values(ascending=False)
        print("Variance:")
        print(self.variance_df)

        if is_removal:
            non_numeric_cols = self.X_train.select_dtypes(exclude=np.number).columns.tolist()
            allowed_cols = self.variance_df[self.variance_df > threshold].index.tolist() + non_numeric_cols
            dropped_cols = [x for x in self.X_train.columns if x not in allowed_cols]
            self.X_train = self.X_train[allowed_cols]
            self.X_test = self.X_test[allowed_cols]
            print("Dropped Columns:")
            print(dropped_cols)

    def extreme_values(self, is_impute=True, threshold=.999):
        """
        min and max for each numeric columns in train df
        :param is_impute: if columns need to be imputed based on extreme values
        :param threshold: threshold cutoff for extreme values
        """
        self.extreme_values_df = self.X_train.select_dtypes(include=np.number).agg(['min', 'max'])
        print("Extreme Values:")
        print(self.extreme_values_df)
        if is_impute:
            for col in self.X_train.select_dtypes(include=np.number).columns:
                percentiles = self.X_train[col].quantile([1-threshold, threshold]).values
                self.X_train[col] = np.clip(self.X_train[col], percentiles[0], percentiles[1])

    def missing_values(self, is_removal: bool = True, threshold: float = 0.99):
        """
        method to calculate % of missing values in each column; drop columns with missing value > threshold
        :param is_removal: if columns need to be dropped based on missing value %
        :param threshold: threshold cutoff for missing values
        """
        self.missing_values_df = self.X_train.isna().mean().sort_values(ascending=False)
        print("Missing Value %:")
        print(self.missing_values_df)

        if is_removal:
            allowed_cols = self.missing_values_df[self.missing_values_df < threshold].index.tolist()
            dropped_cols = [x for x in self.X_train.columns if x not in allowed_cols]
            self.X_train = self.X_train[allowed_cols]
            self.X_test = self.X_test[allowed_cols]
            print("Dropped Columns:")
            print(dropped_cols)

    def check_high_cardinality(self, is_removal: bool = True, threshold: float = 0.9):
        """
        method to check high cardinal variables
        :param is_removal: if columns need to be dropped based on cardinality values
        :param threshold: threshold cutoff for cardinality values
        """
        numeric_columns = self.X_train.select_dtypes(
            include=np.number).columns
        non_numeric_columns = self.X_train.select_dtypes(
            exclude=np.number).columns
        df = self.X_train[non_numeric_columns]
        self.cardinality_df = df.nunique() / df.shape[0]
        self.cardinality_df = self.cardinality_df.sort_values(ascending=False)
        print("Cardinality:")
        print(self.cardinality_df)

        if is_removal:
            allowed_cols = self.cardinality_df[self.cardinality_df < threshold].index.tolist()
            allowed_cols.extend(numeric_columns)
            dropped_cols = [x for x in self.X_train.columns if x not in allowed_cols]
            self.X_train = self.X_train[allowed_cols]
            self.X_test = self.X_test[allowed_cols]
            print("Dropped Columns:")
            print(dropped_cols)

    def numeric_numeric_corr(self, is_removal: bool = True, cutoff_corr: float = 0.7):
        """
        method to find correlation between numeric columns
        :param is_removal: if columns need to be dropped based on correlation values
        :param cutoff_corr: threshold cutoff for correlation values
        """
        numeric_columns = self.X_train.select_dtypes(
            include=np.number).columns.tolist()
        self.numeric_numeric_corr_df = self.X_train[numeric_columns].corr().abs()
        print("Correlations:")
        print(self.numeric_numeric_corr_df)

        if is_removal:
            upper = self.numeric_numeric_corr_df.where(np.triu(np.ones(self.numeric_numeric_corr_df.shape),
                                                               k=1).astype(np.bool))
            to_drop = [column for column in upper.columns if any(upper[column] > cutoff_corr)]
            print("Dropped columns based on high correlation:")
            print(to_drop)
            self.X_train = self.X_train.drop(to_drop, axis=1)
            self.X_test = self.X_test.drop(to_drop, axis=1)
