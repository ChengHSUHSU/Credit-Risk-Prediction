



'''
expect : 
output : X(dataframe),Y(array)
goal : classsification / regression
format unit
'''

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from util import boxplot_for_feature_to_target, hist_for_feature_to_target




class Load_Data:
    def __init__(self):
        a = 0
    def forward(self, plot_feature_to_target=False):
        credit_data = fetch_openml('credit-g', version='active')
        categories_feature = list(credit_data.categories.keys())
        credit_dataframe = pd.DataFrame(credit_data.data, columns=credit_data.feature_names)
        target_dataframe = pd.DataFrame(credit_data.target,columns=['label'])
        target_dataframe_replace_with_onehot = target_dataframe.replace('good',1).replace('bad',0)
        if plot_feature_to_target is True:
            for feature_name in credit_data.feature_names:
                if feature_name in categories_feature:
                    # categorical feature
                    hist_for_feature_to_target(credit_dataframe, target_dataframe,feature_name)
                else:
                    # numeric feature
                    boxplot_for_feature_to_target(credit_dataframe, target_dataframe,feature_name)
        return credit_dataframe, target_dataframe_replace_with_onehot, categories_feature
        

