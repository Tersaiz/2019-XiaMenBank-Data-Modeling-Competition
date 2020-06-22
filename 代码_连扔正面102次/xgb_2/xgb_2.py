from gensim.models import Word2Vec
from collections import defaultdict
from glove import Corpus, Glove
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import catboost as ctb
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold,KFold
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def w2v_transform(X,word2vec,length):
    length = len(base_col[3:])
    return np.array([np.hstack([
            np.mean([word2vec[w] 
                     for w in words if w in word2vec] or
                    [np.zeros(length)], axis=1)
        ,   np.max([word2vec[w] 
                     for w in words if w in word2vec] or
                    [np.zeros(length)], axis=1)
                ])   for words in X
        
        ])



def get_w2v(data_frame,feat,length):
    model = Word2Vec(data_frame[feat].values, size=length, window=20, min_count=1,
                     workers=10, iter=10)
    return model
def w2v_feat(data):
    tr_w2v = get_w2v(data[['rid']],'rid',50)
    vect = w2v_transform(data.rid.values,tr_w2v.wv,50)
    for i in range(vect.shape[1]):
        data['w2vn'+str(i)] = vect[:,i]
    return data



def g2v_transform(X,tr_g2v,length):
    tr_g2v_dict = tr_g2v.dictionary.keys()
    word2vec = tr_g2v.word_vectors
    length = len(base_col[3:])

    return   np.array([np.hstack([
            np.mean([word2vec[tr_g2v.dictionary[w]  ] 
                     for w in words if w in tr_g2v_dict] or
                    [np.zeros(length)], axis=1)
        ,   np.max([word2vec[tr_g2v.dictionary[w]] 
                     for w in words if w in tr_g2v_dict] or
                    [np.zeros(length)], axis=1)
        
        ])   for words in X
        
        ])
    
def g2v_feat(data):
    def glove_feat(df, feat, length):
        corpus = Corpus() 
        corpus.fit(df[feat], window=20)
        glove = Glove(no_components=length, learning_rate=0.05)

        glove.fit(corpus.matrix, epochs=10, no_threads=10, verbose=True)
        glove.add_dictionary(corpus.dictionary)

        return glove
    tr_g2v = glove_feat(data[['rid']],'rid',50)
    vect = g2v_transform(data.rid.values,tr_g2v,50)

    for i in range(vect.shape[1]):
        data['g2v'+str(i)] = vect[:,i]
    return data

def get_time(train):
    train['dt_bg'] = train.certValidBegin-2208988800
    train['dt_bg'] = pd.to_datetime(train['dt_bg'] * 1000000000)
    train['dt_year'] = train.dt_bg.dt.year
    train['dt_month'] = train.dt_bg.dt.year*100+train.dt_bg.dt.month
    #train['dt_st'] = train.certValidStop-2208988800
    #train['dt_st'] = pd.to_datetime(train['dt_st'] * 1000000000)
    #train['dt_year1'] = train.dt_st.dt.year
    train['bk_time'] = (train['age']*365*24*3600+ train.certValidBegin-all_data.certValidBegin.max())/365/24/3600
    del train['dt_bg']
    return train

train = pd.read_csv('../new_data/train.csv') #.query('isNew==1').reset_index(drop=True)
target = pd.read_csv('../new_data/train_target.csv')
train = train.merge(target,on='id',how='left')
train1 = train.query('isNew==0').reset_index(drop=True)

train1['residentAddr']-=300000
train1.loc[train1['residentAddr']==-300999,'residentAddr'] = -999
train = train.query('isNew==1').reset_index(drop=True)

test = pd.read_csv('../new_data/test.csv')
all_data = train.append(test).reset_index(drop=True)
#sub = pd.read_csv('../submit/test_pred4.csv')
#all_data = all_data.merge(sub,how='left')
# all_data['bankCard1'] = all_data['bankCard'].apply(lambda x: str(x)[:6]).astype(float)
# all_data['dist_res'] = all_data['dist']/all_data['residentAddr']
# all_data['certId_res'] = all_data['certId']/all_data['residentAddr']
# all_data['dist1'] = all_data['dist'].apply(lambda x: str(x)[:2]).astype(float)
# all_data['residentAddr1'] = all_data['residentAddr'].apply(lambda x: str(x)[:2]).astype(float)
# all_data['dist2'] = all_data['dist'].apply(lambda x: str(x)[:3]).astype(float)
# all_data['residentAddr2'] = all_data['residentAddr'].apply(lambda x: str(x)[:3]).astype(float)

train1['bankCard1'] = train1['bankCard'].apply(lambda x: str(x)[:6]).astype(float)
train1['dist_res'] = train1['dist']/all_data['residentAddr']
train1['certId_res'] = train1['certId']/all_data['residentAddr']
train1['dist1'] = train1['dist'].apply(lambda x: str(x)[:2]).astype(float)
train1['residentAddr1'] = train1['residentAddr'].apply(lambda x: str(x)[:2]).astype(float)
train1['dist2'] = train1['dist'].apply(lambda x: str(x)[:3]).astype(float)
train1['residentAddr2'] = train1['residentAddr'].apply(lambda x: str(x)[:3]).astype(float)

all_data['certTime'] = all_data.certValidStop - all_data.certValidBegin
all_data['week_hour'] = all_data['weekday']*100+all_data['setupHour']
all_data = get_time(all_data)  
train1 = get_time(train1)  

zx_col = ['x_'+str(i) for i in range(78)] +['ncloseCreditCard', 'unpayIndvLoan', 'unpayOtherLoan', 'unpayNormalLoan', '5yearBadloan']

tmp = all_data[zx_col].corr()
drop_col = []
base_col = []
for i in zx_col:
    base_col.append(i)
    tmp1 = tmp[i]
    tmp2 = tmp1[tmp1==1].index.tolist()
    tmp2 = [n for n in tmp2 if n not in base_col]
    drop_col.extend(tmp2)
drop_col = list(set(drop_col))
zx_col = [i for i in zx_col if i not in drop_col]


for i in tqdm(zx_col):

    all_data[i+'wk_cnt'] = all_data.groupby(['weekday',i])[i].transform('count')
    if i!='lmt':
        all_data[i+'lmt_mean'] = all_data.groupby([i])['lmt'].transform('mean')
        all_data[i+'lmt_max'] = all_data.groupby([i])['lmt'].transform('max')
    
cat_col = ['dt_year','dt_month','job','basicLevel','ethnic','highestEdu','dist','gender','age','loanProduct', 'lmt', 'bankCard', 'residentAddr', 'linkRela','setupHour', 'weekday']
for i in tqdm(cat_col):
    all_data[i+'_cnt'] = all_data.groupby([i])[i].transform('count')
    tmp = train1.groupby(i,as_index=False)['target'].agg({i+'_cnt1':'count',i+'_sum1':'sum'})
    all_data =all_data.merge(tmp,how='left')
    
    all_data[i+'_cnt1'] = all_data[i+'_cnt'] /(all_data[i+'_cnt1'] +3)

    if i!='weekday':
        all_data[i+'weekday_cnt'] = all_data.groupby(['weekday',i])[i].transform('count')
#        tmp = train1.groupby(['weekday',i],as_index=False)['target'].agg({i+'weekday_cnt1':'count'})
#        all_data =all_data.merge(tmp,how='left')
#        all_data[i+'weekday_cnt1'] = all_data[i+'weekday_cnt'] /(all_data[i+'weekday_cnt1'] +3)
#        del all_data[i+'weekday_cnt']        
        
    if i!='loanProduct':
        all_data[i+'loanProduct_cnt'] = all_data.groupby(['loanProduct',i])[i].transform('count')
        tmp = train1.groupby(['loanProduct',i],as_index=False)['target'].agg({i+'loanProduct_cnt1':'count',i+'loanProduct_mn1':'mean'})
        all_data =all_data.merge(tmp,how='left')
        all_data[i+'loanProduct_cnt1'] = all_data[i+'loanProduct_cnt'] /(all_data[i+'loanProduct_cnt1'] +3)
        del all_data[i+'loanProduct_cnt']
    if i!='residentAddr':
        all_data[i+'residentAddr_cnt'] = all_data.groupby(['residentAddr',i])[i].transform('count')
        tmp = train1.groupby(['residentAddr',i],as_index=False)['target'].agg({i+'residentAddr_cnt1':'count'})
        all_data =all_data.merge(tmp,how='left')
        all_data[i+'residentAddr_cnt1'] = all_data[i+'residentAddr_cnt'] /(all_data[i+'residentAddr_cnt1'] +3)
        del all_data[i+'residentAddr_cnt']        
#        if i!='loanProduct':
#            all_data[i+'residentAddrloanProduct_cnt'] = all_data.groupby(['loanProduct','residentAddr',i])[i].transform('count')
#            tmp = train1.groupby(['loanProduct','residentAddr',i],as_index=False)['target'].agg({i+'residentAddrloanProduct_cnt1':'count'})
#            all_data =all_data.merge(tmp,how='left')
#            all_data[i+'residentAddrloanProduct_cnt1'] = all_data[i+'residentAddrloanProduct_cnt'] /(all_data[i+'residentAddrloanProduct_cnt1'] +3)
#            del all_data[i+'residentAddrloanProduct_cnt1']
#     if i!='linkRela':
#         all_data[i+'linkRela_cnt'] = all_data.groupby(['linkRela',i])[i].transform('count')
#         tmp = train1.groupby(['linkRela',i],as_index=False)['target'].agg({i+'linkRela_cnt1':'count'})
#         all_data =all_data.merge(tmp,how='left')
#         all_data[i+'linkRela_cnt1'] = all_data[i+'linkRela_cnt'] /(all_data[i+'linkRela_cnt1'] +3)
#         del all_data[i+'linkRela_cnt']     


unique_col = 'loanProduct,lmt,basicLevel,bankCard,residentAddr,linkRela,setupHour,weekday'.split(',')
for i,col in enumerate(unique_col[:-1]):
    for j,col1 in enumerate(unique_col[i+1:]):
        all_data[col+col1+'_nunique'] = all_data.groupby([col])[col1].transform('nunique')
        tmp = train1.groupby(col,as_index=False)[col1].agg({col+col1+'_nunique1':'nunique'})
        all_data =all_data.merge(tmp,how='left')    
        all_data[col+col1+'_nunique1'] = all_data[col+col1+'_nunique'] /(all_data[col+col1+'_nunique1'] +3)


all_data['age_lmt_mn'] = all_data.groupby(['age'])['lmt'].transform('mean')
tmp = all_data.groupby(['age'],as_index=False)['lmt'].agg({'age_lmt_mn1':'mean'})
all_data =all_data.merge(tmp,how='left')
all_data['age_lmt_mn1'] = all_data['age_lmt_mn']/all_data['age_lmt_mn1']


# all_data['loanProduct_basicLevel_max'] = all_data.groupby(['loanProduct'])['basicLevel'].transform('max')
# all_data['id_cnt'] =   all_data.groupby(['certId','certValidStop','certValidBegin','residentAddr','gender','loanProduct'])['target'].transform('count')    
# tmp = train1.groupby(['certId','certValidStop','certValidBegin','residentAddr','gender','loanProduct'],as_index=False)['target'].agg({'tr0_mean':'mean','tr0_count':'count'})  
# all_data = all_data.merge(tmp,how='left')
all_data['rid'] = all_data.apply(lambda x: [ i+'x'+str(x[i]) for i in zx_col  ],axis=1) 
all_data = w2v_feat(all_data)
all_data = g2v_feat(all_data)
del all_data['rid']    

all_data = pd.get_dummies(all_data,columns=['loanProduct'])
train = all_data[~all_data.target.isnull()]
test = all_data[all_data.target.isnull()]

y = train['target']

feat_col = list(train)
feat_col.remove('id')
feat_col.remove('target')
feat_col.remove('certId')
feat_col.remove('dist')


oof = np.zeros(len(train))
pred = np.zeros(len(test))
score_list = []
feat_imp = pd.DataFrame(feat_col,columns=['feat'])
for i in range(5):
    skf = StratifiedKFold(n_splits=5, random_state=2019*i+i, shuffle=True)

    for index, (train_index, test_index) in enumerate(skf.split(train, y)):
            #print('fold : ',index,train.shape)
            train_x, test_x, train_y, test_y = train.iloc[train_index], train.iloc[test_index], y.iloc[train_index], y.iloc[test_index]
            train_x = train_x[feat_col]
            train_x = train_x.append(train1.query('target==1'))[feat_col]
            train_y = train_y.append(train1.query('target==1').target)
            test_x = test_x[feat_col]
            print(train_x.shape)                  
            xgb_model = xgb.XGBClassifier( learning_rate=0.01, n_estimators=10000, max_depth=6 ,
                 tree_method = 'gpu_hist',subsample=0.9, colsample_bytree=0.7, min_child_samples=5,eval_metric = 'auc',random_state=88
                )
            xgb_model.fit(train_x, train_y, eval_set=[(train_x, train_y),(test_x, test_y)],early_stopping_rounds=500, verbose=None)  
            oof[test_index] = xgb_model.predict_proba(test_x)[:,1]
            pred+=xgb_model.predict_proba(test[feat_col])[:,1]
            score_list.append(xgb_model.best_score)
            feat_imp['skf'+str(index)] = xgb_model.feature_importances_
    print(score_list)
    print('train auc : ',roc_auc_score(y,oof))
    print('train mean auc : ',np.mean(score_list))
    print([i for i in oof if i==0])
    
sub = test[['id']]
sub['target'] = pred/25
print(sub['target'].round().value_counts())
sub.to_csv('result/xgb_3.csv',index=None)


'''
100%|██████████| 49/49 [00:08<00:00,  5.66it/s]
100%|██████████| 16/16 [00:23<00:00,  1.46s/it]
Performing 10 training epochs with 10 threads
Epoch 0
Epoch 1
Epoch 2
Epoch 3
Epoch 4
Epoch 5
Epoch 6
Epoch 7
Epoch 8
Epoch 9
(38790, 617)
(38790, 617)
(38790, 617)
(38791, 617)
(38791, 617)
[0.72859, 0.753618, 0.761355, 0.734364, 0.741215]
train auc :  0.7241587379650822
train mean auc :  0.7438284000000001
[]
(38790, 617)
(38790, 617)
(38790, 617)
(38791, 617)
(38791, 617)
[0.72859, 0.753618, 0.761355, 0.734364, 0.741215, 0.722282, 0.729504, 0.766021, 0.746929, 0.750648]
train auc :  0.7412784767580765
train mean auc :  0.7434526
[]
(38790, 617)
(38790, 617)
(38790, 617)
(38791, 617)
(38791, 617)
[0.72859, 0.753618, 0.761355, 0.734364, 0.741215, 0.722282, 0.729504, 0.766021, 0.746929, 0.750648, 0.765459, 0.72675, 0.739428, 0.751911, 0.735175]
train auc :  0.7277981320323424
train mean auc :  0.7435499333333334
[]
(38790, 617)
(38790, 617)
(38790, 617)
(38791, 617)
(38791, 617)
[0.72859, 0.753618, 0.761355, 0.734364, 0.741215, 0.722282, 0.729504, 0.766021, 0.746929, 0.750648, 0.765459, 0.72675, 0.739428, 0.751911, 0.735175, 0.738255, 0.729303, 0.748574, 0.726091, 0.757181]
train auc :  0.726174997441572
train mean auc :  0.74263265
[]
(38790, 617)
(38790, 617)
(38790, 617)
(38791, 617)
(38791, 617)
[0.72859, 0.753618, 0.761355, 0.734364, 0.741215, 0.722282, 0.729504, 0.766021, 0.746929, 0.750648, 0.765459, 0.72675, 0.739428, 0.751911, 0.735175, 0.738255, 0.729303, 0.748574, 0.726091, 0.757181, 0.766262, 0.722621, 0.681306, 0.785703, 0.739784]
train auc :  0.7183987102613089
train mean auc :  0.74193316
[]
0.0    23561
Name: target, dtype: int64
'''
