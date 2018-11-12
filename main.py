# -*- coding: utf-8 -*-
# @Time    : 2018/11/12 23:12
# @Author  : quincyqiang
# @File    : main.py
# @Software: PyCharm

import gc # 垃圾回收
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb
pd.set_option('display.max_columns',100)
gc.enable()



df = pd.read_csv('./input/bank-additional-train.csv')
df['y'].replace(['no','yes'],[0,1],inplace=True)

def add_poly_features(data,column_names):
    # 组合特征
    features=data[column_names]
    rest_features=data.drop(column_names,axis=1)
    poly_transformer=PolynomialFeatures(degree=2,interaction_only=False,include_bias=False)
    poly_features=pd.DataFrame(poly_transformer.fit_transform(features),columns=poly_transformer.get_feature_names(column_names))

    for col in poly_features.columns:
        rest_features.insert(1,col,poly_features[col])
    return rest_features


def create_feature(df):
    # 数值型数据处理
    # 'age', 'balance', 'duration', 'campaign', 'pdays', 'previous'，emp.var.rate，cons.price.idx，cons.conf.idx，euribor3m，nr.employed 
    def standardize_nan(x):
        # 标准化
        x_mean = np.nanmean(x)  # 求平均值，但是个数不包括nan
        x_std = np.nanstd(x)
        return (x - x_mean) / x_std
        # 対数変換

    ## 直方图绘制时分布不均匀的特征
    df['log_age'] = np.log(df['age'])
    df['log_std_age'] = standardize_nan(df['log_age'])
    # df["log_duration"] = np.log(df['duration']+ 1) # duration 字段不能用
    df["log_campaign"] = np.log(df['campaign'] + 1)
    df["log_pdays"] = np.log(df['pdays'] - df['pdays'].min() + 1)
    df['log_previous'] = np.log(df['previous'] + 1)  # 这里没有+1
    df = df.drop(["age", "duration", "campaign", "pdays", "previous"], axis=1)  # duration 字段不能用

    df['log_emp.var.rate'] = np.log(df['emp.var.rate'] + 1)  # 这里没有+1
    df['log_cons.price.idx'] = np.log(df['cons.price.idx'] + 1)  # 这里没有+1
    df['log_euribor3m'] = np.log(df['euribor3m'] + 1)  # 这里没有+1
    df['log_nr.employed '] = np.log(df['nr.employed'] + 1)  # 这里没有+1
    df = df.drop(["emp.var.rate", "cons.price.idx", "euribor3m", "nr.employed"], axis=1)

    # month 文字列与数値的変換
    # month 文字列与数値的変換
    df['month'] = df['month'].map({'jan': 1,
                                           'feb': 2,
                                           'mar': 3,
                                           'apr': 4,
                                           'may': 5,
                                           'jun': 6,
                                           'jul': 7,
                                           'aug': 8,
                                           'sep': 9,
                                           'oct': 10,
                                           'nov': 11,
                                           'dec': 12
                                           }).astype(int)
    # 1月:0、2月:31、3月:(31+28)、4月:(31+28+31)、 ...
    day_sum = pd.Series(np.cumsum([0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]), index=np.arange(1, 13))
    df['date'] = (df['month'].map(day_sum)).astype(int)
    # ------------End 数据预处理 类别编码-------------

    # ---------- Start 数据预处理 类别型数据------------
    # 类别型数据
    # cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'day_of_week', 'month','poutcome']
    cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month','poutcome']
    df.drop('day_of_week',axis=1,inplace=True)
    df = pd.get_dummies(df, columns=cate_cols)
    # ------------End 数据预处理 类别编码----------
    cols = [col for col in df.columns if col not in ['y']]
    train_len=18000
    new_train, new_test = df[:train_len], df[train_len:]
    return new_train,new_test


# 调整参数
def tune_params(model,params,X,y):
    gsearch = GridSearchCV(estimator=model,param_grid=params, scoring='roc_auc')
    gsearch.fit(X, y)
    print(gsearch.cv_results_, gsearch.best_params_, gsearch.best_score_)
    return gsearch


# 特征重要性
def plot_fea_importance(classifier,X_train):
    plt.figure(figsize=(10,12))
    name = "xgb"
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],
                    x=classifier.feature_importances_[indices][:40],orient='h')
    g.set_xlabel("Relative importance", fontsize=12)
    g.set_ylabel("Features", fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " feature importance")
    plt.show()


def evaluate_cv5_lgb(train_df, test_df, cols, test=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    y_test = 0
    oof_train = np.zeros((train_df.shape[0],))
    for i, (train_index, val_index) in enumerate(kf.split(train_df[cols])):
        X_train, y_train = train_df.loc[train_index, cols], train_df.y.values[train_index]
        X_val, y_val = train_df.loc[val_index, cols], train_df.y.values[val_index]
        xgb = XGBClassifier(n_estimators=4000,
                            learning_rate=0.03,
                            num_leaves=30,
                            colsample_bytree=.8,
                            subsample=.9,
                            max_depth=7,
                            reg_alpha=.1,
                            reg_lambda=.1,
                            min_split_gain=.01,
                            min_child_weight=2,
                            verbose=True)
        xgb.fit(X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                early_stopping_rounds=100, eval_metric=['auc'], verbose=True)
        y_pred = xgb.predict_proba(X_val)[:,1]
        if test:
            y_test += xgb.predict_proba(test_df.loc[:, cols])[:,1]
        oof_train[val_index] = y_pred
        if i==0:
            plot_fea_importance(xgb,X_train)
    gc.collect()
    auc = roc_auc_score(train_df.y.values, oof_train)
    y_test /= 5
    print('5 Fold auc:', auc)
    return y_test


train,test=create_feature(df)
cols = [col for col in train.columns if col not in ['id','y']]
y_pred=evaluate_cv5_lgb(train,test,cols,True)
print(roc_auc_score(test.y,y_pred))