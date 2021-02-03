


'''
1/31
data load module
data process module
=> add standardize for lr / svm (3)
model module
=> add logistic regression[DONE] / svm / decision tree[DONE] / random forest[DONE] / adaboost[DONE] /  gbdt[DONE] / xgboost[DONE]
=> add cross validation (4)
evaluation module
=> add eval metric such as accuracy / precision / recall / F1 [DONE]
=> add eval plot such ROC (5)
=> add explainable route (6)
'''

from load_data import Load_Data
from data_process import Data_Processing
from model import Model
from evaluation import Evaluation

if __name__ == '__main__':
    # load data
    load_data_module = Load_Data()
    credit_dataframe, target_dataframe, categories_feature = load_data_module.forward()
    # data processing
    data_processing_module = Data_Processing(credit_dataframe, target_dataframe, categories_feature)
    train_credit_dataframe, train_target_dataframe, test_credit_dataframe, test_target_dataframe = \
        data_processing_module.forward(used_model = 'xgboost')
    # modeling
    model_module = Model(train_credit_dataframe, train_target_dataframe, categories_feature, used_model = 'xgboost')
    model = model_module.forward()
    # evaluating
    evaluation_module  = Evaluation(model, test_credit_dataframe, test_target_dataframe, categories_feature, used_model = 'xgboost')
    evaluation_module.forward()











