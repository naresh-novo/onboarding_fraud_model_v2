from os import stat
from xml.etree.ElementTree import QName
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import plotly.express as px

import warnings

warnings.simplefilter("ignore")

class Explainer :

    def __init__(self) -> None:
        pass

    def feature_importance(self,model, X, imp_type='gain'):
        '''
        Returns the best binary classifier with tuned hyperparameter set

            Parameters:
                    model (object): model object of the binary classifier
                    X (dataframe(pandas)): pandas dataframe of predictor variables in train dataset
                    imp_type (str): importance type to be plotted from the model, choose from ['gain', 'cover', 'weight', 'total_gain', 'total_cover']
            
            Prints:
                    feat_importances (plot): plots the feature importance

            Returns:
                    feat_importance (dataframe(pandas)): pandas dataframe of feature importances

            Raises:
                    ValueError: If X is not pandas dataframe
        '''
        # Check if train dataset is passed as dataframe
        if type(X) != pd.core.frame.DataFrame:
            raise ValueError('Train dataset not passed as pandas dataframe')
        
        # Check if the classifier is tree based
        if type(model).__name__ in (['DecisionTreeClassifier', 'RandomForestClassifier', 'XGBClassifier']):
            importance_array = np.fromiter(model.get_booster().get_score(importance_type=imp_type).values(), dtype=float)
        else:
            importance_array = model.coef_[0]
        
        # Extract the feature importance
        feat_importances = pd.DataFrame(importance_array, index=X.columns, columns=['importance'])

        # Plot the feature importance
        fig = px.bar(feat_importances, orientation='h', text_auto='.2s')
        fig.update_layout(yaxis={'categoryorder':'total ascending'}, width=1300, height=800)
        fig.show()

        # Return the feature importance dataframe
        return feat_importances