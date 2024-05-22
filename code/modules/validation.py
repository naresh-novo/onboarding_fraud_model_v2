from os import stat
from xml.etree.ElementTree import QName
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import plotly.express as px
from sklearn.model_selection import train_test_split
from modules.modeling import ModelMetric
import warnings

warnings.simplefilter("ignore")

mm = ModelMetric()

class validator:

    def __init__(self) -> None:
        pass

    def rank_order_test(self,df, selected_features,dependent_variable, modelx, seeds):
        X = df[selected_features]
        y = df[dependent_variable]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seeds)

        modelx.fit(X_train, y_train)
        print(modelx.score(X_test,y_test))

        model_train_dict = mm.model_metrics(modelx.predict(X_train),y_train,modelx.predict_proba(X_train),'Train')
        model_test_dict = mm.model_metrics(modelx.predict(X_test),y_test,modelx.predict_proba(X_test),'Test')

        
        return model_train_dict,model_test_dict
        #return(res1[res1.columns[:2]],res2[res2.columns[:2]])