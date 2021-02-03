import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import xgboost as xgb
from util import one_DF_split_to_two_DF, predict_score, test_predict_score, metric_vs_hyp_plot
from evaluation import Evaluation


class Model:
    def __init__(self, train_val_credit_dataframe, train_val_target_dataframe, categories_feature, used_model):
        self.used_model = used_model
        if self.used_model == 'light':
            self.train_credit_dataframe, self.train_target_dataframe, self.val_credit_dataframe, self.val_target_dataframe = \
                one_DF_split_to_two_DF(train_val_credit_dataframe, train_val_target_dataframe, split_rate=0.875)
            self.train_val_credit_dataframe, self.train_val_target_dataframe = train_val_credit_dataframe, train_val_target_dataframe
            self.categories_feature = categories_feature
            # hyperparameter setting
            self.params = { \
                'boosting' : 'gbdt', 
                'objective' : 'binary', #or binary | cross_entropy
                'metric' : ['auc', 'average_precision', 'cross_entropy'],
                'learning_rate' : 0.3, # default=0.1
                'lambda_l1' : 0.0, # default=0.0
                'lambda_l2' : 1, # default=0.0
                'max_depth' : 6, # default=-1
                'num_leaves' : 31, # default=31  
            }
            self.num_round = 10
        
        elif self.used_model == 'xgboost' :
            # convert into array format
            self.X_train_array = train_val_credit_dataframe.to_numpy()
            self.Y_train_array = train_val_target_dataframe.to_numpy()
            self.params = \
                {
                    'objective': 'binary:logistic',
                    'eta' : 0.3, #eta = learning rate | default=0.3
                    'alpha' : 0, #alpha = lambda_l1 | default=0
                    'lambda' : 1, #lambda = lambda_l2 | default=1
                    'max_depth': 6, # default=6, where max_depth=0 -> free growing
                }
            self.num_round = 10
            self.train_data = xgb.DMatrix(self.X_train_array, label=self.Y_train_array)
        
        elif self.used_model == 'lr' or self.used_model == 'dt' or \
             self.used_model == 'rf' or self.used_model == 'gbdt' or \
             self.used_model == 'adaboost': 
            # convert into array format
            self.X_train_array = train_val_credit_dataframe.to_numpy()
            self.Y_train_array = train_val_target_dataframe.to_numpy()
            # tree based model parameter setting
            self.max_depth = 10 #decision-tree #random-forest #gbdt
            self.random_state = 0 #random-forest #gbdt #adaboost
            self.n_estimators = 100 #gbdt #adaboost
            self.learning_rate = 1.0 #gbdt

    def hyperparameter_plot(self):
        '''
        tuning hyp:
        learning_rate (0.05~0.3 interval=0.025)
        , lambda_l1, lambda_l2, (0.0125~0.1 interval=0.0125)
        max_depth, (5~80 interval=5)
        num_leaves (15~50 interval=5)
        '''
        show_learning_rate = [3,5,7,9,11,13,15]
        # accuracy
        train_acc_scores, val_acc_scores = list(), list()
        # precision
        train_prec_scores, val_prec_scores = list(), list()
        # recall
        train_rec_scores, val_rec_scores = list(), list()
        for hyp in show_learning_rate:
            train_data = lgb.Dataset(data=self.train_credit_dataframe,
                                     label=self.train_target_dataframe,
                                     categorical_feature=self.categories_feature)
            tuning_hyp = 'num_round'
            if tuning_hyp != 'num_round':
                self.params[tuning_hyp] = hyp
                model = lgb.train(self.params, train_data, self.num_round)
            else:
                model = lgb.train(self.params, train_data, hyp)
            train_pred_result = model.predict(self.train_credit_dataframe)
            val_pred_result = model.predict(self.val_credit_dataframe)
            # accuracy
            train_accuracy = test_predict_score(train_pred_result, self.train_target_dataframe['label'], metric='accuracy')
            val_accuracy = test_predict_score(val_pred_result, self.val_target_dataframe['label'], metric='accuracy')
            train_acc_scores.append(train_accuracy)
            val_acc_scores.append(val_accuracy)            
            # precision
            train_precision = test_predict_score(train_pred_result, self.train_target_dataframe['label'], metric='precision')
            val_precision = test_predict_score(val_pred_result, self.val_target_dataframe['label'], metric='precision')
            train_prec_scores.append(train_precision)
            val_prec_scores.append(val_precision)
            # recall
            train_recall = test_predict_score(train_pred_result, self.train_target_dataframe['label'], metric='recall')
            val_recall = test_predict_score(val_pred_result, self.val_target_dataframe['label'], metric='recall')
            train_rec_scores.append(train_recall)
            val_rec_scores.append(val_recall)
        # accuracy
        metric_vs_hyp_plot(show_learning_rate, train_acc_scores, val_acc_scores, hyp_name=tuning_hyp, metric_name='Accuracy')
        # precision
        metric_vs_hyp_plot(show_learning_rate, train_prec_scores, val_prec_scores, hyp_name=tuning_hyp, metric_name='Precision')
        # recall
        metric_vs_hyp_plot(show_learning_rate, train_rec_scores, val_rec_scores, hyp_name=tuning_hyp, metric_name='Recall')

    def forward(self, hyperparameter_plot_show=False):
        if self.used_model == 'light' : 
            if hyperparameter_plot_show is True:
                self.hyperparameter_plot()
            train_data = lgb.Dataset(data=self.train_val_credit_dataframe,
                                     label=self.train_val_target_dataframe,
                                     categorical_feature=self.categories_feature)
            gbm = lgb.train(self.params, train_data, self.num_round)
            return gbm
        
        elif self.used_model == 'lr' : 
            logistic_regression = LogisticRegression().fit(self.X_train_array, self.Y_train_array)
            return logistic_regression
        
        elif self.used_model == 'dt' :
            dtree = DecisionTreeClassifier(max_depth=self.max_depth).fit(self.X_train_array,self.Y_train_array)
            return dtree
        
        elif self.used_model == 'rf' : 
            rforest = DecisionTreeClassifier(max_depth=self.max_depth,
                                             random_state=self.random_state).fit(self.X_train_array,self.Y_train_array)
            return rforest
        
        elif self.used_model == 'gbdt' :
            gbdt = GradientBoostingClassifier(n_estimators=self.n_estimators,
                                              learning_rate=self.learning_rate,
                                              max_depth=self.max_depth,
                                              random_state=self.random_state).fit(self.X_train_array,self.Y_train_array)
            return gbdt
        
        elif self.used_model == 'adaboost' :
            adaboost = AdaBoostClassifier(n_estimators=self.n_estimators,
                                          random_state=self.random_state).fit(self.X_train_array,self.Y_train_array)
            return adaboost
        
        elif self.used_model == 'xgboost' :
            xgboost = xgb.train(self.params, self.train_data, self.num_round)
            return xgboost

