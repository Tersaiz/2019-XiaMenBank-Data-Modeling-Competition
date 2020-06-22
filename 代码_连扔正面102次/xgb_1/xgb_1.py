#%%
import pandas as pd
from pandas.core.frame import DataFrame
import numpy as np
from sklearn import metrics
# from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif,chi2
from sklearn.feature_selection import SelectKBest, SelectPercentile
from data_preprocess import dataPreprocess
import xgboost as xgb
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from feature_selector import FeatureSelector
import lightgbm as lgb
from sklearn.ensemble import *
from sklearn.tree import *
from sklearn.svm import *
from xgboost import *

#输入数据、标签以及测试数据路径
data_path = "../../new_data/train.csv"
label_path = "../../new_data/train_target.csv"
predict_data_path = "../../new_data/test.csv"

#得到输入数据、标签及测试数据
df = pd.read_csv(data_path)
df_label  = pd.read_csv(label_path)
df_predict  = pd.read_csv(predict_data_path)

#得到输入数据的标签
column_headers = list( df.columns.values )

#使用数据的均值、中位数或众数来填充缺失数据
def impute_NA_with_avg(data, strategy='mean', NA_col=[]):
    """
    replacing the NA with mean/median/most frequent values of that variable.
    Note it should only be performed over training set and then propagated to test set.
    """
    data_copy = data.copy(deep=True)
    for i in NA_col:
        if data_copy[i].isnull().sum() > 0:
            if strategy == 'mean':
                data_copy[i + '_impute_mean'] = data_copy[i].fillna(data[i].mean())
            elif strategy == 'median':
                data_copy[i + '_impute_median'] = data_copy[i].fillna(data[i].median())
            elif strategy == 'mode':
                data_copy[i + '_impute_mode'] = data_copy[i].fillna(data[i].mode()[0])
        else:
            print("Column %s has no missing" % i)
    return data_copy
#得到训练数据集，将训练数据与数据标签对应
train_data = pd.merge(df, df_label, how='left', on=['id'])

# 如果数据中全为缺失值则将该数据删除
train_data.dropna(how='all')

# 使用均值来对训练集和测试集中缺失值进行填充
train_data = impute_NA_with_avg(train_data, column_headers)
df_predict = impute_NA_with_avg(df_predict, column_headers)

#在后面导出数据时使用
id_list = df_predict["id"].tolist()

#得到标签为1和标签为0的样本，后续选择验证样本时使用
train_data_1=train_data[train_data['target']==1]
train_data_0=train_data[train_data['target']==0]

#得到训练数据标签
y = train_data["target"]
x = train_data.drop(['isNew','target'],axis=1)

#观察数据中residentAddr的编码方式，重新构造了特征
df.loc[df['isNew'] ==0,'residentAddr'] = df[df['isNew'] ==0]['residentAddr'].apply(lambda x: x if x == -999 else x-300000)

#特征选择，特征选择的参数解释：

"""
missing_threshold表示数据特征缺失值比例阈值，当缺失值比例超过0.6时则删除该特征
correlation_threshold表示特征之间的相关性
task指的是进行的任何，eval_metric表示使用的评价指标
cumulative_importance指的是按特征重要性排序后的特征累加，看多少个特征重要性累加可以达到0.95
"""

fs = FeatureSelector(data=x, labels=y)
fs.identify_all(selection_params={'missing_threshold': 0.6, 'correlation_threshold': 0.9,
                                  'task': 'regression', 'eval_metric': 'mse',
                                  'cumulative_importance': 0.95})

choose = fs.remove(methods=['missing', 'single_unique', 'zero_importance'], keep_one_hot=True)

#根据选择得到的特征集来得到训练数据和测试数据集
x = x[choose.columns.values]
X_predict = df_predict[choose.columns.values]
choose.columns

#因为存在样本不均衡问题，因而在选择测试数据集时，将50%为1的样本选做测试集
label_1=train_data_1['target']
label_0=train_data_0['target']
train_data_1=train_data_1[choose.columns.values]
train_data_0=train_data_0[choose.columns.values]
X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(train_data_0, label_0, test_size=.2, random_state=333)
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(train_data_1, label_1, test_size=.5, random_state=333)

X_test=pd.concat([X_test_0,X_test_1],axis=0)
y_test=pd.concat([y_test_0,y_test_1],axis=0)

#训练模型，考虑到数据样本不均衡，使用所有的数据进行训练，并且正负样本给予不同的权重
xgboost= xgb.XGBClassifier(max_depth=3,learning_rate=0.01,n_estimators=1000,scale_pos_weight=12)
xgboost_model = xgboost.fit(x, y,eval_set=[(x,y),(X_test,y_test)],eval_metric='auc',
                            early_stopping_rounds=30,verbose = 100)

#进行特征重要性排序
importances = xgboost_model.feature_importances_
indices = np.argsort(importances)[::-1]
feat_labels = x.columns
print("Feature ranking:")
#    l1,l2,l3,l4 = [],[],[],[]
for f in range(x.shape[1]):
    print("%d. feature no:%d feature name:%s (%f)" % (
    f + 1, indices[f], feat_labels[indices[f]], importances[indices[f]]))
print(">>>>>", importances)

#得到预测结果值，做后续导出时使用
y_pred = xgboost_model.predict(X_test)
y_predprob = xgboost_model.predict_proba(X_test)[:, 1]

print("Accuracy : %.4g" % metrics.accuracy_score(y_test.values, y_pred))  # Accuracy : 0.9852
print("AUC Score (Train): %f" % metrics.roc_auc_score(y_test, y_predprob))
print("precision_score (Train): %f" % metrics.precision_score(y_test, y_pred))
print("recall_score (Train): %f" % metrics.recall_score(y_test, y_pred))
print("f1_score (Train): %f" % metrics.f1_score(y_test, y_pred))

#导出数据
y_pp = xgboost_model.predict_proba(X_predict)[:, 1]
c = {"target": y_pp}
data_lable = DataFrame(c)  # 将字典转换成为数据框

d = {"id": id_list, "target": y_pp}
res = DataFrame(d)  # 将字典转换成为数据框

res.to_csv("result/jinrong_xgb_1.csv",index = False)

