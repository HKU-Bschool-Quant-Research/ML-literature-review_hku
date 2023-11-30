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
# import sklearn.preprocessing as sp
# import sklearn.pipeline as pl
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
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

def PLS(X_train,Y_train,X_test,n_component=2):
    model = PLSRegression(n_components=n_component)
    model.fit(X_train,Y_train)
    Y_predict=model.predict(X_test)
    return Y_predict

#Generalized linear model (k-term spline series)   
# def Glm_lasso(X_train,Y_train,X_test,alpha=0.5,K_trem=2,max_iter=1000):
#     model=pl.make_pipeline(sp.PolynomialFeatures(K_trem), Lasso(alpha,max_iter=max_iter))
#     model.fit(X_train,Y_train)
#     Y_predict=model.predict(X_test)
#     return Y_predict

def GBrt(X_train,Y_train,X_test,max_depth=2,n_trees=100,learning_rate=0.1,Huber_robust=False): #following xiu's paper, tree hyperparameters need to tune, include maximum depth, number of trees in ensemble and shrinking shrinkage weight(learning rate)  
    model = xgb.XGBRegressor(max_depth=max_depth,reg_lambda=1,n_estimators=n_trees,learning_rate=learning_rate,random_state=1) #reg_lambda=1 means training with l2 impurity. 
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    return Y_predict

def GBrt_H(X_train,Y_train,X_test,max_depth=2,n_trees=100,learning_rate=0.1,Huber_robust=False): #following xiu's paper, tree hyperparameters need to tune, include maximum depth, number of trees in ensemble and shrinking shrinkage weight(learning rate)  
    model = xgb.XGBRegressor(objective="reg:pseudohubererror",max_depth=max_depth,reg_lambda=1,n_estimators=n_trees,learning_rate=learning_rate,random_state=1) #reg_lambda=1 means training with l2 impurity. 
    model.fit(X_train, Y_train)
    Y_predict = model.predict(X_test)
    return Y_predict

def rf(X_train,Y_train,X_test,max_depth=6): #Depth L of the trees and number of bootstrap samples B are the tuning parameters optimized via validation
    model = RandomForestRegressor(max_depth=max_depth,n_estimators=300,max_features="sqrt",random_state=1)
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
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=10000,callbacks=[callback])
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
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=10000,callbacks=[callback])
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
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=10000,callbacks=[callback])
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
    model.fit(X_train,Y_train,validation_data=(X_validation,Y_validation),epochs=100,batch_size=10000,callbacks=[callback])
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
    model.fit(X_train,Y_train,epochs=100,validation_data=(X_validation,Y_validation),batch_size=10000,callbacks=[callback])
    Y_predict=model.predict(X_test)

    return Y_predict






##long-short zero-net-investment portfolio  
###should input a panel data with columns of shocks id and rows of date, covering y prediction.    
def profit_calculate(Ypredict,Y_test,decile=10):
    total_number=len(Ypredict.columns)
    decile_number=total_number//decile  ##call numbers
    profit_list=pd.DataFrame(np.zeros((len(Ypredict.index),11)),columns=["low(L)","2","3","4","5","6","7","8","9","high(H)","H-L"])
        
    for i in range(len(Ypredict.index)):
        produce_list=Ypredict.iloc[i].sort_values(ascending=1) #ascending sort
        
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

    annual_return=collect.sum()*12/len()
    
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
            DR_value=collect.loc[i,j]
            if DR_value <0:
                DR_values.loc[i,j]=DR_value
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
        for i in range(month_numbers):
            accumulate+=collect.loc[i,j]
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

 #########
#main body#
 #########
 
#chunk_size=100000
#for chunk in pd.read_csv("datashare.csv",chunksize=chunk_size):
 ################################   
#testing sample of coding process#
 ################################
from sklearn.model_selection import train_test_split    
#import data
data_row= pd.read_csv("datashare.csv",nrows=5000)
factor_name=data_row.columns
# from sklearn.model_selection import train_test_split

##Data clearning (drop data that missing value of y, and fill x with means)
data=data_row.iloc[:,:7]
data=data.dropna(subset=list(data_row.columns)[6],inplace=False)
for i in range(7):
    data.iloc[:,i].fillna(value=data.iloc[:,i].mean(),inplace=True)
X=data.iloc[:,:6]
y=data.iloc[:,[0,1,6]]
X_train_row, X_test_row, Y_train_row, Y_test_row = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_row, X_validation_row, Y_train_row, Y_validation_row = train_test_split(X_train_row, Y_train_row, test_size=0.2, random_state=0)

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

##extract factor
X_train=X_train_row.iloc[:,2:6]
Y_train=Y_train_row.iloc[:,2]
X_validation=X_validation_row.iloc[:,2:6]
Y_validation=Y_validation_row.iloc[:,2]
X_test=X_test_row.iloc[:,2:6]


# get the prediction of different models
y_test_pred_ols=Ols(X_train, Y_train, X_test,Huber_robust=True)
y_test_pred_ols_H=Ols(X_train, Y_train, X_test,Huber_robust=True)
y_test_pred_pcr=Pcr(X_train, Y_train, X_test)
y_test_pred_pcr_H=Pcr(X_train, Y_train, X_test,Huber_robust=True)
y_test_pred_pls=PLS(X_train, Y_train, X_test,n_component=2)
y_test_pred_gbrt=GBrt(X_train,Y_train,X_test,max_depth=2,n_trees=100,learning_rate=0.1,Huber_robust=False)
y_test_pred_gbrt_H=GBrt(X_train,Y_train,X_test,max_depth=2,n_trees=100,learning_rate=0.1,Huber_robust=True)
y_test_pred_rf=rf(X_train,Y_train,X_test,max_depth=6)
y_test_pred_nn1=nn1(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn2=nn2(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn3=nn3(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn4=nn4(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)
y_test_pred_nn5=nn5(X_train,Y_train,X_test,X_validation,Y_validation,l1=0.001,lr=0.01,random_seed=1)

############ uncomplete ######### 
##convert to panel data taht columns of shocks id and rows of date, covering y prediction
Y_test_row=Y_test_row.set_index(["DATE","permno"])[Y_test_row.columns[2]]
Y_test_row=Y_test_row.unstack()



y_test_pred_ols=pd.concat([Y_test_row.iloc[:,:2],Ols(X_train, Y_train, X_test)],axis=1)

y_test_pred_ols=y_test_pred_ols.set_index(["DATE","permno"])[y_test_pred_ols.columns[2]]
y_test_pred_ols=y_test_pred_ols.unstack()



# #

profit_ols=profit_calculate(y_test_pred_ols, Y_test_row)

l=["OLS","OLS_H"]
collect_np=np.concatenate([y_test_pred_ols,y_test_pred_ols_H],axis=1)
collect=pd.DataFrame(collect_np,colunms=l)
# y_predict=GBrt_H(X_train,Y_train,X_test)