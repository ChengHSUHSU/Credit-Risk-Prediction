import random
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score, precision_score ,recall_score
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def one_DF_split_to_two_DF(dataframe1, dataframe2, split_rate=0.8):
    sample_index_shuffle = random.sample([i for i in range(dataframe1.shape[0])],k=dataframe1.shape[0])
    train_sample_index = sample_index_shuffle[:int(len(sample_index_shuffle) * split_rate)]
    test_sample_index = sample_index_shuffle[int(len(sample_index_shuffle) * split_rate):]
    train_dataframe1 = dataframe1.iloc[train_sample_index]
    test_dataframe1 = dataframe1.iloc[test_sample_index]
    train_dataframe2 = dataframe2.iloc[train_sample_index]
    test_dataframe2 = dataframe2.iloc[test_sample_index]
    return train_dataframe1, train_dataframe2, test_dataframe1, test_dataframe2


def DF_categorical_feature_to_OneHot(dataframe, categorical_feature):
    for feature in categorical_feature:
        dataframe[feature] = dataframe[feature].astype(int)
    dataframe = pd.get_dummies(dataframe,columns=categorical_feature)
    return dataframe



def predict_score(pred_prob, Y_test_array, binary_threshold=0.5):
    pred_one_hot = list()
    for i in range(pred_prob.shape[0]):
        if pred_prob[i] >= binary_threshold:
            pred_one_hot.append(1)
        else:
            pred_one_hot.append(0)
    print(metrics.classification_report(list(Y_test_array), pred_one_hot))
    print('---------------------------------------')
    print('Confusion Matrix')
    print(np.transpose(confusion_matrix(list(Y_test_array), pred_one_hot).T))
    print('---------------------------------------')
    print('positive label : 0 | negative label : 1')


def test_predict_score(pred_prob, Y_test_array, metric='accuracy', binary_threshold=0.5):
    pred_one_hot = list()
    for i in range(pred_prob.shape[0]):
        if pred_prob[i] >= binary_threshold:
            pred_one_hot.append(1)
        else:
            pred_one_hot.append(0)
    if metric == 'accuracy':
        return accuracy_score(list(1 - Y_test_array), list(1 - np.array(pred_one_hot)))
    elif metric == 'precision':
        return precision_score(list(1 - Y_test_array), list(1 - np.array(pred_one_hot)))
    elif metric == 'recall':
        return recall_score(list(1 - Y_test_array), list(1 - np.array(pred_one_hot)))
    else:
        print('this is undefined metric.....')



def metric_vs_hyp_plot(hyp_list, train_scores, test_scores, hyp_name='xxx-hyp',metric_name='xxx-metric'):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.set_xlabel(hyp_name)
    ax.set_ylabel(metric_name)
    ax.set_title(metric_name + ' vs ' + hyp_name + 'for train and val sets')
    ax.plot(hyp_list, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(hyp_list, test_scores, marker='o', label="val",
            drawstyle="steps-post")
    ax.legend()
    plt.savefig('plot/model_layer/'+hyp_name+'-'+metric_name+'.png')
    plt.clf()


def boxplot_for_feature_to_target(feature_dataframe, target_dataframe, numeric_feature='credit_amount'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_target_dataframe = pd.concat([feature_dataframe, target_dataframe],axis=1)
    ax = sns.boxplot(x="label", y=numeric_feature, data=feature_target_dataframe)
    plt.savefig('plot/feature_layer/'+'boxplot'+'-'+numeric_feature + '.png')
    plt.clf()



def hist_for_feature_to_target(feature_dataframe, target_dataframe, category_feature='checking_status'):
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    feature_target_dataframe = pd.concat([feature_dataframe, target_dataframe],axis=1).astype(int, errors='ignore')
    ax = sns.histplot(data=feature_target_dataframe, x=category_feature, hue="label", multiple="dodge", shrink=.8)
    plt.savefig('plot/feature_layer/'+ 'histplot' + '-' + category_feature + '.png')
    plt.clf()
