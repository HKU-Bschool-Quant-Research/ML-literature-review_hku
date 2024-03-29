# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# !pip install pandas
# !pip install numpy
# !pip install statsmodels
# !pip install sklearn
# !pip install gc
# !pip install xgboost
import pandas as pd
import numpy as np
import statsmodels.api as sm
import os  
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
# import sklearn.preprocessing as sp
# import sklearn.pipeline as pl
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
import tensorflow as tf
from tensorflow.keras import regularizers

import math
import gc
new_dir="D:\\replication\\datashare"
os.chdir(new_dir)

 #############
#data separate#
 #############
def preprocess(rdata,validation_year,test_year):
    ##in dataset, the ytpe of date is numpy.int64, such as 19570131
    # validation_year=19700000  #1970_1987
    # test_year=19870000
    train_len=0
    validation_len =0
    test_len=0
    for i in range(len(rdata.index)):
        if rdata.iloc[i,1] < validation_year:
            train_len+=1
        elif rdata.iloc[i,1] >validation_year and rdata.iloc[i,1] < test_year:
            validation_len+=1
    test_len=len(rdata.index)-train_len-validation_len
    
    #separate dataset     
    train_data=rdata.iloc[:train_len,]
    validation_data=rdata.iloc[train_len:train_len+validation_len,]
    test_data=rdata.iloc[train_len+validation_len:train_len+validation_len+test_len,]
    return train_data,validation_data,test_data

 ##########
#algorithms#
 ##########
#OLS banchmark
def Ols(X_train,Y_train,X_test,Huber_robust=False): # Huber_robust ->True
    model = sm.OLS(Y_train, X_train)
    if Huber_robust:
        model = model.fit(cov_type='HC3')
    else: 
        model = model.fit()
    Y_predict=model.predict(X_test)
    return Y_predict
  
#dimension reduction models  
def Pcr(X_train,Y_train,X_test,n_component="mle",Huber_robust=False):
    pca=PCA(n_components=n_component)
    fit=pca.fit(X_train)
    X_train_convert=fit.transform(X_train)
    X_test_convert=fit.transform(X_test)
    Y_predict=Ols(X_train_convert,Y_train,X_test_convert,Huber_robust)
    return Y_predict

def PLS(X_train,Y_train,X_test,n_component=15):
    model = PLSRegression(n_components=n_component)
    model.fit(X_train,Y_train)
    Y_predict=model.predict(X_test)
    return Y_predict


def Ela(X_train, y_train, X_test, loss='huber', penalty='elasticnet', alpha=0.1, l1_ratio=0.5):
    model = SGDRegressor(loss=loss, penalty=penalty, alpha=alpha, l1_ratio=l1_ratio)
    model.fit(X_train, y_train)
    Y_predict=model.predict(X_test)
    return Y_predict

#Generalized linear model (two-term spline series)   
# def Glm(X_train, y_train, X_test):
#     knots = np.linspace(0, 1, 5)
#     bspline = sm.splines.bspline(X_train.flatten(), knots=knots, degree=2)
#     X_train_bspline = bspline.design_matrix
#     X_test_bspline = bspline(X_test.flatten())
#     model = sm.GLM(y_train, X_train_bspline, family=sm.families.HuberT())
#     result = model.fit()
#     Y_predict = result.predict(X_test_bspline)
#     return Y_predict

#Generalized linear model (k-term spline series)   
# def Glm_lasso(X_train,Y_train,X_test,alpha=0.5,K_trem=2,max_iter=1000):
#     model=pl.make_pipeline(sp.PolynomialFeatures(K_trem), Lasso(alpha,max_iter=max_iter))
#     model.fit(X_train,Y_train)
#     Y_predict=model.predict(X_test)
#     return Y_predict

# def spline_basis(X, knots):
#     basis_functions = []
#     for knot in knots:
#         basis_functions.append((X - knot) ** 2)
#     basis_functions.insert(0, np.ones_like(X))  # 添加常数项1
#     basis_functions.insert(1, X)  # 添加一次项z
#     return np.concatenate(basis_functions, axis=1)

# def Glm_lasso(X_train,Y_train,X_test,alpha=0.1,rou=0.5): # knots=5
#     knots = [0, 1, 2, 3, 4]
#     X_train_basis = spline_basis(X_train,knots)
#     X_test_basis = spline_basis(X_test,knots)
    
#     # 使用带有Elastic Net惩罚的SGDRegressor拟合广义线性模型
#     alpha = 0.1  # Group Lasso的正则化参数
#     l1_ratio = 0.5  # Elastic Net的L1和L2的混合比例
#     model = SGDRegressor(penalty='elasticnet', alpha=alpha, l1_ratio=l1_ratio, max_iter=1000, tol=1e-4)
#     model.fit(X_train_basis, y_train)
#     # 获取估计的回归系数
#     theta_estimate = model.coef_
#     # 使用估计的回归系数进行预测
#     Y_prediction =np.dot(X_test_basis, theta_estimate)
#     return Y_prediction


def GBrt(X_train,Y_train,X_test,max_depth=2,n_trees=500,learning_rate=0.1,Huber_robust=False,X_validation='None',Y_validation='None'): #following xiu's paper, tree hyperparameters need to tune, include maximum depth, number of trees in ensemble and shrinking shrinkage weight(learning rate)  
    if type(X_validation)==type("None"):
        model = xgb.XGBRegressor(max_depth=max_depth,reg_lambda=1,n_estimators=n_trees,learning_rate=learning_rate,random_state=1) #reg_lambda=1 means training with l2 impurity. 
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
    else:
        t=0
        best_entree=n_trees
        best_lr=learning_rate
        for n_trees in range(100,1000,200): ##
            for learning_rate in range(0.1,1,0.1):
                model = xgb.XGBRegressor(max_depth=max_depth,reg_lambda=1,n_estimators=n_trees,learning_rate=learning_rate,random_state=1) #reg_lambda=1 means training with l2 impurity. 
                model.fit(X_train, Y_train)
                Y_pred_val = model.predict(X_validation)
                r2_sc=r2_score(Y_validation, Y_pred_val)
                t+=1
                if t==1:
                    r2score=r2_sc                
                if r2_sc>=r2score:
                    r2score=r2_sc
                    best_entree= n_trees
                    best_lr= learning_rate
        X_train=pd.concat([X_train,X_validation],axis=0)
        Y_train=pd.concat([Y_train,Y_validation],axis=0)     
        model = xgb.XGBRegressor(max_depth=max_depth,reg_lambda=1,n_estimators=best_entree,learning_rate=best_lr,random_state=1) #reg_lambda=1 means training with l2 impurity. 
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
    return Y_predict

def GBrt_H(X_train,Y_train,X_test,max_depth=2,n_trees=500,learning_rate=0.1,Huber_robust=False,X_validation='None',Y_validation='None'): #following xiu's paper, tree hyperparameters need to tune, include maximum depth, number of trees in ensemble and shrinking shrinkage weight(learning rate)  
    if type(X_validation)==type("None"):
        model = xgb.XGBRegressor(objective="reg:pseudohubererror",max_depth=max_depth,reg_lambda=1,n_estimators=n_trees,learning_rate=learning_rate,random_state=1) #reg_lambda=1 means training with l2 impurity. 
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
    else:
        t=0
        best_entree=n_trees
        best_lr=learning_rate
        for n_trees in range(100,1000,200): ##
            for learning_rate in range(0.1,1,0.1):
                model = xgb.XGBRegressor(objective="reg:pseudohubererror",max_depth=max_depth,reg_lambda=1,n_estimators=n_trees,learning_rate=learning_rate,random_state=1) #reg_lambda=1 means training with l2 impurity. 
                model.fit(X_train, Y_train)
                Y_pred_val = model.predict(X_validation)
                r2_sc=r2_score(Y_validation, Y_pred_val)
                t+=1
                if t==1:
                    r2score=r2_sc
                if r2_sc>=r2score:
                    r2score=r2_sc
                    best_entree= n_trees
                    best_lr= learning_rate
        X_train=pd.concat([X_train,X_validation],axis=0)
        Y_train=pd.concat([Y_train,Y_validation],axis=0)     
        model = xgb.XGBRegressor(objective="reg:pseudohubererror",max_depth=max_depth,reg_lambda=1,n_estimators=best_entree,learning_rate=best_lr,random_state=1) #reg_lambda=1 means training with l2 impurity. 
        model.fit(X_train, Y_train)
        Y_predict = model.predict(X_test)
    return Y_predict


def rf(X_train,Y_train,X_test,max_depth=6,n_estimator=300,X_validation='None',Y_validation='None'): #Depth L of the trees and number of bootstrap samples B are the tuning parameters optimized via validation
    if type(X_validation)==type("None"):
        model=RandomForestRegressor(max_depth=max_depth,n_estimators=n_estimator,max_features="sqrt",random_state=1)
        model.fit(X_train, Y_train)    
        Y_prediction = model.predict(X_test)
    else:
        t=0
        best_estimators=100
        for nest in range(100,1000,100):      
            model=RandomForestRegressor(max_depth=max_depth,n_estimators=nest,max_features="sqrt",random_state=1)
            model.fit(X_train, Y_train)
            Y_pred_val=model.predict(X_validation)
            r2_sc=r2_score(Y_validation, Y_pred_val)
            t+=1
            if t==1:
               r2score=r2_sc
            if r2_sc>r2score:
                r2score=r2_sc
                best_estimators=nest
                
        X_train=pd.concat([X_train,X_validation],axis=0)
        Y_train=pd.concat([Y_train,Y_validation],axis=0)
        model=RandomForestRegressor(max_depth=max_depth,n_estimators=best_estimators,max_features="sqrt",random_state=1)
        model.fit(X_train, Y_train)    
        Y_prediction = model.predict(X_test)
    return Y_prediction

 #############
#neuro netwrok#
 #############
# activaion: Relu, l1 penalty, epoch 100, barch size 10000, early stopping with "loss" and patience 5.
def nn1(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1):
    tf.random.set_seed(random_seed)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l1(l1),activation="relu",input_shape=(len(X_train.columns),)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation="relu"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss="mean_squared_error")
    callback=tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5)
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=5000,callbacks=[callback])
    Y_predict=model.predict(X_test)
    return Y_predict

def nn2(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1):
    tf.random.set_seed(random_seed)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l1(l1),activation="relu",input_shape=(len(X_train.columns),)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation="relu"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss="mean_squared_error")
    callback=tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5)
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=5000,callbacks=[callback])
    Y_predict=model.predict(X_test)
    return Y_predict


def nn3(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1):
    tf.random.set_seed(random_seed)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l1(l1),activation="relu",input_shape=(len(X_train.columns),)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(8,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation="relu"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss="mean_squared_error")
    callback=tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5)
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=5000,callbacks=[callback])
    Y_predict=model.predict(X_test)
    return Y_predict

def nn4(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1):
    tf.random.set_seed(random_seed)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l1(l1),activation="relu",input_shape=(len(X_train.columns),)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(8,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation="relu"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss="mean_squared_error")
    callback=tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5)
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=5000,callbacks=[callback])
    Y_predict=model.predict(X_test)
    return Y_predict

def nn5(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1):
    tf.random.set_seed(random_seed)
    model=tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(32,kernel_regularizer=regularizers.l1(l1),activation="relu",input_shape=(len(X_train.columns),)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(8,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(4,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(2,kernel_regularizer=regularizers.l1(l1),activation="relu"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(1,activation="relu"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss="mean_squared_error")
    callback=tf.keras.callbacks.EarlyStopping(monitor="loss",patience=5)
    model.fit(X_train,Y_train,epochs=100,validation_data=(X_validation,Y_validation),batch_size=5000,callbacks=[callback])
    Y_predict=model.predict(X_test)

    return Y_predict






##long-short zero-net-investment portfolio  
###should input a panel data with columns of shocks id and rows of date, covering y prediction.    
def profit_calculate(Ypredict,Y_test,decile=10):
    total_number=len(Ypredict.columns)
    profit_list=pd.DataFrame(np.zeros((len(Ypredict.index),11)),columns=["low(L)","2","3","4","5","6","7","8","9","high(H)","H-L"])
        
    for i in range(len(Ypredict.index)):
        produce_list=Ypredict.iloc[i].dropna(inplace=False).sort_values(ascending=1) #ascending sort
        decile_number=len(produce_list)//decile ##call numbers
        
        list_low=produce_list.index.values[:decile_number] # low
        list_2=produce_list.index.values[decile_number:decile_number*2]
        list_3=produce_list.index.values[decile_number*2:decile_number*3]
        list_4=produce_list.index.values[decile_number*3:decile_number*4]
        list_5=produce_list.index.values[decile_number*4:decile_number*5]
        list_6=produce_list.index.values[decile_number*5:decile_number*6]
        list_7=produce_list.index.values[decile_number*6:decile_number*7]
        list_8=produce_list.index.values[decile_number*7:decile_number*8]
        list_9=produce_list.index.values[decile_number*8:decile_number*9]
        list_high=produce_list.index.values[-decile_number:] #high
        
# average weight
        date=Y_test.index[i]
        profit_list.iloc[i,0]=(Y_test.loc[date,list_low].sum()/decile_number) 
        profit_list.iloc[i,1]=(Y_test.loc[date,list_2].sum()/decile_number)
        profit_list.iloc[i,2]=(Y_test.loc[date,list_3].sum()/decile_number)
        profit_list.iloc[i,3]=(Y_test.loc[date,list_4].sum()/decile_number)
        profit_list.iloc[i,4]=(Y_test.loc[date,list_5].sum()/decile_number)
        profit_list.iloc[i,5]=(Y_test.loc[date,list_6].sum()/decile_number)
        profit_list.iloc[i,6]=(Y_test.loc[date,list_7].sum()/decile_number)
        profit_list.iloc[i,7]=(Y_test.loc[date,list_8].sum()/decile_number)
        profit_list.iloc[i,8]=(Y_test.loc[date,list_9].sum()/decile_number)
        profit_list.iloc[i,9]=(Y_test.loc[date,list_high].sum()/decile_number)
        profit_list.iloc[i,10]=(Y_test.loc[date,list_high].sum()/decile_number)-(Y_test.loc[date,list_low].sum()/decile_number) 
  ############      
# predict list# Temporarily vacant 
  ############    

    return profit_list
    

#performance evaluation
##include annual_return,annual_volatility,sharp ratio,downside risk, Sortino ratio,maximum downside decrease and MPPM.
## input a dataframes with columns of different models profit result and rows of date.
def perf_evaluation(collect):
    month_numbers=len(collect.index)
    column_name=list(collect.columns)
    #annual_return

    annual_return=collect.sum()*12/month_numbers
    
    #annual_volatility
    annual_volatility=collect.std()*(12**0.5)
    
    #sharp ratio
    annual_shape=(annual_return)/annual_volatility
    
    #downside risk
    DR_values=pd.DataFrame(np.zeros((month_numbers,len(column_name))))
    DR_values.columns=column_name
    for j in column_name:
        for i in range(month_numbers):
            #DR_value=collect.loc[i,j]-average_month[j]
            date= collect.index[i]
            DR_value=collect.loc[date,j]
            if DR_value <0:
                DR_values.loc[date,j]=DR_value
    annual_DR=DR_values.std()*(12**0.5)
    #Sortino ratio
    
    #annual_Sortino(annual_return devided by downside risk)
    annual_Sortino=(annual_return)/annual_DR
    
    #maximum downside decrease
    MDD=pd.Series(np.arange(len(column_name)),index=column_name,dtype="float64")
    for j in column_name:
        accumulate=0
        temporary_max=0
        mdd=[]
        accumulate_list=[]
        max_time=1
        for i in range(month_numbers):
            date= collect.index[i]
            accumulate+=collect.loc[date,j]
            accumulate_list.append(accumulate)
            acc_max=max(accumulate_list)
            if accumulate==acc_max:
                temporary_max=accumulate
                max_time=i
            if accumulate==min(accumulate_list[max_time:]):
                mdd.append(temporary_max-accumulate)
        MDD[j]=max(mdd)
        
    #MPPM
    gamma=4 # risk aversion factor
    MPPM=pd.Series(np.arange(len(column_name)),index=column_name,dtype="float64")
    for j in column_name:
        mppm=(1/(1-gamma))*math.log((1/month_numbers)*np.sum(((collect[j]/100)+1)**(1-gamma)))
        MPPM[j]=mppm*100*12
    
    performance=pd.concat([annual_return,annual_volatility,annual_shape,annual_DR,annual_Sortino,MDD,MPPM],axis=1)
    performance_name=['annual_return','annual_volatility(%)','annual_shape_ratio','annual_DR(%)','annual_Sortino_ratio','MDD(%)','MPPM(%)']
    performance.columns=performance_name
    return performance

##variable importance
def VI_cal(X_train,Y_train,X_validation,Y_validation,model_list):
    VI_list=pd.DataFrame(np.zeros((len(X_train.columns),len(model_list))),columns=model_list,index=X_train.columns)
    R_square_vi=pd.DataFrame(np.zeros((len(X_train.columns),len(model_list))),columns=model_list,index=X_train.columns)
    lis=list(X_train.columns)
    lis.append('full train')
    for i in lis:
        if i != "full train":
            X_train_vi=X_train.drop(i,axis=1)
        else:
            X_train_vi=X_train
        y_test_pred_ols=Ols(X_train_vi, Y_train, X_train_vi,Huber_robust=False)
        y_test_pred_ols_H=Ols(X_train_vi, Y_train, X_train_vi,Huber_robust=True)
        y_test_pred_pcr=Pcr(X_train_vi, Y_train, X_train_vi)
        y_test_pred_pcr_H=Pcr(X_train_vi, Y_train, X_train_vi,Huber_robust=True)
        y_test_pred_pls=PLS(X_train_vi, Y_train, X_train_vi,n_component=2)
        y_test_pred_ela_H=Ela(X_train_vi, Y_train, X_train_vi)
        y_test_pred_gbrt=GBrt(X_train_vi,Y_train,X_train_vi,max_depth=2,n_trees=100,learning_rate=0.1,Huber_robust=False)
        y_test_pred_gbrt_H=GBrt(X_train_vi,Y_train,X_train_vi,max_depth=2,n_trees=100,learning_rate=0.1,Huber_robust=True)
        y_test_pred_rf=rf(X_train_vi,Y_train,X_train_vi,max_depth=6)
        y_test_pred_nn1=nn1(X_train_vi,Y_train,X_train_vi,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
        y_test_pred_nn2=nn2(X_train_vi,Y_train,X_train_vi,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
        y_test_pred_nn3=nn3(X_train_vi,Y_train,X_train_vi,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
        y_test_pred_nn4=nn4(X_train_vi,Y_train,X_train_vi,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
        y_test_pred_nn5=nn5(X_train_vi,Y_train,X_train_vi,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)

        for j in model_list:
            R_square_vi.loc[i,j]=r2_score(Y_train, eval("y_test_pred_"+j))
            
    R_square=R_square_vi.iloc[-1,:]-R_square_vi.iloc[:-1,:]
    for j in list(VI_list.columns):
        VI_list.loc[:,j]=(R_square.loc[:,j]-R_square.loc[:,j].min())/(R_square.loc[:,j].max()-R_square.loc[:,j].min())
    return VI_list


 #########
#main body#
 #########
 
#chunk_size=100000
#for chunk in pd.read_csv("datashare.csv",chunksize=chunk_size):
    
    
 ################################   
#testing sample of coding process#
 ################################
# from sklearn.model_selection import train_test_split    
#import data
nrow=10000
data_row= pd.read_csv("chars60_rank_2000s.csv",nrows=nrow)
data_row=data_row.drop(['sic','gvkey'],axis=1) #delete lable of stock indicator and redundant industry label
#nrow=len(data_row.index)

#conver normal variable to dummy variant
ffi49_dummy= pd.get_dummies(data_row['ffi49'],drop_first=False,prefix='ffi49')
year_dummy= pd.get_dummies(data_row['year'],drop_first=False,prefix='year')
data_row=data_row.drop(['ffi49','year'],axis=1)
data=pd.concat([ffi49_dummy,data_row],axis=1)
#data=data_row
data['date']=pd.to_datetime(data['date']).dt.strftime("%Y%m%d").astype(int)
data['ret']=data['ret']*100
data.sort_values(by='date',ascending=True,inplace=True)
data.reset_index()
#factor_name=data_row.columns

# from sklearn.model_selection import train_test_split

##Data clearning (drop data that missing value of y, and fill x with means)
# data=data_row.iloc[:,:7]
# data=data.dropna(subset=list(data_row.columns)[6],inplace=False)
# for i in range(7):
#     data.iloc[:,i].fillna(value=data.iloc[:,i].mean(),inplace=True)
y=data[['permno','date','ret']]
X=data.drop('ret',axis=1)


## 60% training set, 20% validation set and 20% test set
train_set_number=int(nrow*0.6)
val_set_number=test_set_number=int(nrow*0.2)

X_train_row=X.iloc[:train_set_number,:]
X_validation_row=X.iloc[train_set_number:train_set_number+val_set_number,:]
X_test_row=X.iloc[train_set_number+val_set_number:,:]

Y_train_row=y.iloc[:train_set_number,:]
Y_validation_row=y.iloc[train_set_number:train_set_number+val_set_number,:]
Y_test_row=y.iloc[train_set_number+val_set_number:,:]
# X_train_row, X_test_row, Y_train_row, Y_test_row = train_test_split(X, y, test_size=0.2, random_state=0)
# X_train_row, X_validation_row, Y_train_row, Y_validation_row = train_test_split(X_train_row, Y_train_row, test_size=0.2, random_state=0)

 ################
#replication code#
 ###############
#based on dataset of xiu's paper, however, the data pubilshed by xiu is incomplete, 
#which lacks return of each shock, that is, y of the model, so here only show the framework of 
#main body, and need to completed it after get specific data

# rdata = pd.read_csv("datashare.csv")
# validation_year=19700000
# test_year=19870000
# y_label="ret"
# Y_name=["permno","date","ret"]
# train_data,validation_data,test_data=preprocess(rdata,validation_year,test_year)
# Y_train_row=train_data.loc[:,Y_name] ## stock id, date and return
# X_train_row=train_data.drop(y_label,axis=1)

# #free up memory space
# del train_data
# gc.collect()

# Y_validation_row=validation_data.loc[:,Y_name] ## stock id, date and return
# X_validation_row=validation_data.drop(y_label,axis=1)
# del validation_data
# gc.collect()

# Y_test_row=test_data.loc[:,Y_name] ## stock id, date and return
# X_test_row=test_data.drop(y_label,axis=1)
# del test_data
# gc.collect()

#extract factor
X_train=X_train_row.drop(['permno','date'],axis=1)
Y_train=Y_train_row.drop(['permno','date'],axis=1)
X_validation=X_validation_row.drop(['permno','date'],axis=1)
Y_validation=Y_validation_row.drop(['permno','date'],axis=1)
X_test=X_test_row.drop(['permno','date'],axis=1)
Y_test=Y_test_row.drop(['permno','date'],axis=1)














    
# get the prediction of different models
y_test_pred_ols=Ols(X_train, Y_train, X_test,Huber_robust=False)
y_test_pred_ols_H=Ols(X_train, Y_train, X_test,Huber_robust=True)
y_test_pred_pcr=Pcr(X_train, Y_train, X_test)
y_test_pred_pcr_H=Pcr(X_train, Y_train, X_test,Huber_robust=True)
y_test_pred_pls=PLS(X_train, Y_train, X_test,n_component=5)
y_test_pred_ela_H=Ela(X_train, Y_train, X_test)
y_test_pred_gbrt=GBrt(X_train,Y_train,X_test,max_depth=5,n_trees=1000,learning_rate=0.1,Huber_robust=False)
y_test_pred_gbrt_H=GBrt(X_train,Y_train,X_test,max_depth=5,n_trees=1000,learning_rate=0.1,Huber_robust=True)
y_test_pred_rf=rf(X_train,Y_train,X_test,max_depth=6,n_estimator=300,X_validation=X_validation,Y_validation=Y_validation)
y_test_pred_nn1=nn1(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn2=nn2(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn3=nn3(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn4=nn4(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn5=nn5(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)   
   


##convert to panel data taht columns of shocks id and rows of date, covering y prediction
Y_test_fram=Y_test_row.set_index(["date","permno"])[Y_test_row.columns[2]]
Y_test_fram=Y_test_fram.unstack()


# this function convert result of model prediction to panel data
def panel_convert(y_test_pred_model,Y_test_row):
    if type(y_test_pred_model)!= type(Y_test_row.iloc[:,0]):
        y_test_pred_model=pd.DataFrame(y_test_pred_model,index=Y_test_row.index)

    y_test_pred=pd.concat([Y_test_row.iloc[:,:2],y_test_pred_model],axis=1)
    y_test_pred=y_test_pred.set_index(["date","permno"])[y_test_pred.columns[2]]
    y_test_pred=y_test_pred.unstack()
    return y_test_pred


#convert all prediction to panel dataframe
pred_ols=panel_convert(y_test_pred_ols,Y_test_row)
pred_ols_H=panel_convert(y_test_pred_ols_H,Y_test_row)
pred_pcr=panel_convert(y_test_pred_pcr,Y_test_row)
pred_pcr_H=panel_convert(y_test_pred_pcr_H,Y_test_row)
pred_pls=panel_convert(y_test_pred_pls,Y_test_row)
pred_ela_H=panel_convert(y_test_pred_ela_H,Y_test_row)
pred_gbrt=panel_convert(y_test_pred_gbrt,Y_test_row)
pred_gbrt_H=panel_convert(y_test_pred_gbrt_H,Y_test_row)
pred_rf=panel_convert(y_test_pred_rf,Y_test_row)
pred_nn1=panel_convert(y_test_pred_nn1,Y_test_row)
pred_nn2=panel_convert(y_test_pred_nn2,Y_test_row)
pred_nn3=panel_convert(y_test_pred_nn3,Y_test_row)
pred_nn4=panel_convert(y_test_pred_nn4,Y_test_row)
pred_nn5=panel_convert(y_test_pred_nn5,Y_test_row)


#zero_net_investment result and evaluation the performance
def result_input(pred,Y_test_fram):
    profit=profit_calculate(pred, Y_test_fram)
    profit.index=Y_test_fram.index
    #evaluation result
    evaluation=perf_evaluation(profit)

    return  evaluation

#Rsquare evaluation
model_list=["ols","ols_H","pcr","pcr_H","pls","ela_H","gbrt","gbrt_H","rf","nn1","nn2","nn3","nn4","nn5"]
R_square_pred=pd.Series(range(len(model_list)),index=model_list)
for i in model_list:
    R_square_pred[i]=r2_score(Y_test, eval("y_test_pred_%s" %i))



#variable importance
#VI=VI_cal(X_train,Y_train,X_validation,Y_validation,model_list)

##getting evaluation and output into Excel
eval_ols=result_input(pred_ols,Y_test_fram)
eval_ols_H=result_input(pred_ols_H,Y_test_fram)
eval_pcr=result_input(pred_pcr,Y_test_fram)
eval_pcr_H=result_input(pred_pcr_H,Y_test_fram)
eval_pls=result_input(pred_pls,Y_test_fram)
eval_ela_H=result_input(pred_ela_H,Y_test_fram)
eval_gbrt=result_input(pred_gbrt,Y_test_fram)
eval_gbrt_H=result_input(pred_gbrt_H,Y_test_fram)
eval_rf=result_input(pred_rf,Y_test_fram)
eval_nn1=result_input(pred_nn1,Y_test_fram)
eval_nn2=result_input(pred_nn2,Y_test_fram)
eval_nn3=result_input(pred_nn3,Y_test_fram)
eval_nn4=result_input(pred_nn4,Y_test_fram)
eval_nn5=result_input(pred_nn5,Y_test_fram)

eval_total=pd.DataFrame(columns=eval_ols.columns,index=model_list)
for i in model_list:
    eval_total.loc[i]=eval("eval_%s" %i).loc['H-L',:]
#mention: because samples of y valuse for testing code have no negative values, hence DR equal to 0,as well as MDD


writer=pd.ExcelWriter("ml_literature_review_result_20240103.xlsx")
round_number=6
R_square_pred.round(6).to_excel(writer,sheet_name="R2")
#VI.round(6).to_excel(writer,sheet_name="VI") ##to laege
eval_total.round(round_number).to_excel(writer,sheet_name="evaluation")
eval_ols.round(round_number).to_excel(writer,sheet_name="ols")
eval_ols.round(round_number).to_excel(writer,sheet_name="ols_H")
eval_pcr.round(round_number).to_excel(writer,sheet_name="pcr")
eval_pcr_H.round(round_number).to_excel(writer,sheet_name="pcr_H")
eval_pls.round(round_number).to_excel(writer,sheet_name="pls")
eval_ela_H.round(round_number).to_excel(writer,sheet_name="Ela_H")
eval_gbrt.round(round_number).to_excel(writer,sheet_name="gbrt")
eval_gbrt_H.round(round_number).to_excel(writer,sheet_name="gbrt_H")
eval_rf.round(round_number).to_excel(writer,sheet_name="rf")
eval_nn1.round(round_number).to_excel(writer,sheet_name="nn1")
eval_nn2.round(round_number).to_excel(writer,sheet_name="nn2")
eval_nn3.round(round_number).to_excel(writer,sheet_name="nn3")
eval_nn4.round(round_number).to_excel(writer,sheet_name="nn4")
eval_nn5.round(round_number).to_excel(writer,sheet_name="nn5")

writer.save()

