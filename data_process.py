import random
from sklearn.model_selection import train_test_split
from util import one_DF_split_to_two_DF, DF_categorical_feature_to_OneHot




class Data_Processing:
    def __init__(self, credit_dataframe, target_dataframe, categories_feature):
        self.credit_dataframe = credit_dataframe
        self.target_dataframe = target_dataframe
        self.categories_feature = categories_feature
        # parameter setting
        self.train_rate = 0.8

    def forward(self, used_model='lr'):
        if used_model != 'light':
            self.credit_dataframe = DF_categorical_feature_to_OneHot(self.credit_dataframe, self.categories_feature)
        train_credit_dataframe, train_target_dataframe, test_credit_dataframe, test_target_dataframe = \
            one_DF_split_to_two_DF(self.credit_dataframe, self.target_dataframe, split_rate=0.8)
        return train_credit_dataframe, train_target_dataframe, test_credit_dataframe, test_target_dataframe
