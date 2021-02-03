import sklearn.metrics as metrics
from util import predict_score
import numpy as np


class Evaluation:
    def __init__(self, model, test_credit_dataframe, test_target_dataframe, categories_feature, used_model):
        self.model = model
        if used_model == 'light':
            self.test_credit_dataframe = test_credit_dataframe
        elif used_model == 'xgboost':
            import xgboost as xgb
            self.test_credit_dataframe = xgb.DMatrix(test_credit_dataframe.to_numpy())
        else:
            self.test_credit_dataframe = test_credit_dataframe.to_numpy()
        self.Y_test_array = np.array(test_target_dataframe['label'])

    def forward(self):
        pred_X_test = self.model.predict(self.test_credit_dataframe)
        predict_score(pred_X_test, self.Y_test_array)
