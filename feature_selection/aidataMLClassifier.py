import numpy as np
import pandas as pd
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class AiDataMLClassifier(object):
    def __init__(self,train_dataset,train_datameta,test_dataset,test_datameta):
        self.train_dataset=train_dataset
        self.train_datameta=train_datameta
        self.test_dataset=test_dataset
        self.test_datameta=test_datameta
        self.mymodel=None
    def my_predict(self,modelname,dataset):
        preds=modelname.predict(dataset)
        return preds
    def my_predict_prob(self,modelname,dataset):
        preds=modelname.predict_proba(dataset)
        return preds
    def my_logis_classifier(self,must_var={}):
        scaler=StandardScaler()
        train_data=copy.deepcopy(self.train_dataframe)
        test_data=copy.deepcopy(self.test_dataframe)
        x_list=self.train_datameta['num_col']+self.train_datameta['string_col']
        train_data[x_list]=scaler.fit_transform(train_data[x_list])
        x_train=train_data.loc[:,x_list]
        y_train=train_data.loc[:,self.train_datameta['target_col']]
        test_data[x_list]=scaler.fit_transform(test_data[x_list])
        x_test=test_data.loc[:,x_list]
        y_test=test_data.loc[:,self.test_datameta['target_col']]
        if len(must_var)==0:
            model=LogisticRegression(penalty='l2',C='1.2',fit_intercept='True',solver='lbfgs',max_iter='120')
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        else:
            model=LogisticRegression(penalty=must_var['penalty'],C=must_var['C'],fit_intercept=must_var['fit_intercept'],solver=must_var['solver'],max_iter=must_var['max_iter'])
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        self.mymodel=model
        return model,x_train,y_train,train_pred[:,1],x_test,y_test,test_pred[:,1]
    def my_tree_classifier(self,must_var={}):
        x_list=self.train_datameta['num_col']+self.train_datameta['string_col']
        x_train=self.train_dataframe.loc[:,x_list]
        y_train=self.train_dataframe.loc[:,self.train_datameta['target_col']]
        x_test=self.test_dataframe.loc[:,x_list]
        y_test=self.test_dataframe.loc[:,self.test_datameta['target_col']]
        if len(must_var)==0:
            model=RandomForestClassifier(max_depth=12,criterion='gini',splitter='best',min_samples_leaf=3)
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        else:
            model=RandomForestClassifier(max_depth=must_var['max_depth'],criterion=must_var['criterion'],splitter=must_var['splitter'],min_samples_leaf=must_var['min_samples_leaf'])
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        self.mymodel=model
        return model,x_train,y_train,train_pred[:,1],x_test,y_test,test_pred[:,1]
    def my_svm_classifier(self,must_var={}):
        scaler=StandardScaler()
        train_data=copy.deepcopy(self.train_dataframe)
        test_data=copy.deepcopy(self.test_dataframe)
        x_list=self.train_datameta['num_col']+self.train_datameta['string_col']
        train_data[x_list]=scaler.fit_transform(train_data[x_list])
        x_train=train_data.loc[:,x_list]
        y_train=train_data.loc[:,self.train_datameta['target_col']]
        test_data[x_list]=scaler.fit_transform(test_data[x_list])
        x_test=test_data.loc[:,x_list]
        y_test=test_data.loc[:,self.test_datameta['target_col']]
        if len(must_var)==0:
            model=SVC(C=1.2,kernel='rbf',gamma='auto',C=1.2,max_iter=100,tol=1e-3,probability=True)
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        else:
            model=SVC(kernel=must_var['kernel'],gamma=must_var['gamma'],C=must_var['C'],max_iter=must_var['max_iter'],tol=must_var['tol'],probability=True)
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        self.mymodel=model
        return model,x_train,y_train,train_pred[:,1],x_test,y_test,test_pred[:,1]
    def my_forest_classifier(self,must_var={}):
        x_list=self.train_datameta['num_col']+self.train_datameta['string_col']
        x_train=self.train_dataframe.loc[:,x_list]
        y_train=self.train_dataframe.loc[:,self.train_datameta['target_col']]
        x_test=self.test_dataframe.loc[:,x_list]
        y_test=self.test_dataframe.loc[:,self.test_datameta['target_col']]
        if len(must_var)==0:
            model=RandomForestClassifier(n_estimators=150,max_depth=8,criterion='gini',min_samples_leaf=3,n_jobs=1)
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        else:
            model=RandomForestClassifier(n_estimators=must_var['n_estimators'],max_depth=must_var['max_depth'],criterion=must_var['criterion'],min_samples_leaf=must_var['min_samples_leaf'],n_jobs=must_var['n_jobs'])
            model.fit(x_train,y_train)
            train_pred=self.my_predict_prob(model,x_train)
            test_pred=self.my_predict_prob(model,x_test)
        self.mymodel=model
        return model,x_train,y_train,train_pred[:,1],x_test,y_test,test_pred[:,1]
    def my_lgb_classifier(self,must_var={}):
        x_list=self.train_datameta['num_col']+self.train_datameta['string_col']
        x_train=self.train_dataframe.loc[:,x_list]
        y_train=self.train_dataframe.loc[:,self.train_datameta['target_col']]
        x_test=self.test_dataframe.loc[:,x_list]
        y_test=self.test_dataframe.loc[:,self.test_datameta['target_col']]
        new_x_train,new_x_valid,new_y_train,new_y_valid=train_test_split(x_train,y_train,test_size=0.1)
        train_data=lgb.Dataset(new_x_train,label=new_y_train)
        valid_data=lgb.Dataset(new_x_valid,label=new_y_valid,reference=train_data)
        if len(must_var)==0:
            params = {
                'boosting_type': 'gbdt',
                'objective': 'binary',
                'metric':  'auc',
                'num_leaves':50,
                'learning_rate': 0.01,
                'max_depth': 5,
                'min_data_in_leaf': 5,
                'min_gain_to_split': 0.00001,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'reg_lambda': 0.3,
                'bagging_freq': 10,
                }

            model=lgb.train(params,train_data,valid_sets=valid_data,num_boost_round=200)
            train_pred=model.predict(x_train,num_iteration=model.best_iteration)
            test_pred=model.predict(x_test,num_iteration=model.best_iteration)
        else:
            params=copy.deepcopy(must_var)
            del params['num_boost_round']
            model=lgb.train(params,train_data,valid_sets=valid_data,num_boost_round=must_var['num_boost_round'])
            train_pred=model.predict(x_train,num_iteration=model.best_iteration)
            test_pred=model.predict(x_test,num_iteration=model.best_iteration)
        self.mymodel=model
        return model,x_train,y_train,train_pred[:,1],x_test,y_test,test_pred[:,1]
    def my_knn_classifier(self,must_var={}):
        scaler=StandardScaler()
        train_data=copy.deepcopy(self.train_dataframe)
        test_data=copy.deepcopy(self.test_dataframe)
        x_list=self.train_datameta['num_col']+self.train_datameta['string_col']
        train_data[x_list]=scaler.fit_transform(train_data[x_list])
        x_train=train_data.loc[:,x_list]
        y_train=train_data.loc[:,self.train_datameta['target_col']]
        test_data[x_list]=scaler.fit_transform(test_data[x_list])
        x_test=test_data.loc[:,x_list]
        y_test=test_data.loc[:,self.test_datameta['target_col']]
        if len(must_var)==0:
            model=KNeighborsClassifier(n_neighbors=5,weights='uniform',algorithm='auto',leaf_size=30,metric='minkowski',n_jobs=2)
            model.fit(x_train,y_train)
            train_pred=self.my_predict(model,x_train)
            test_pred=self.my_predict(model,x_test)
        else:
            model=KNeighborsClassifier(n_neighbors=must_var['n_neighbors'],weights=must_var['weights'],algorithm=must_var['algorithm'],leaf_size=must_var['leaf_size'],metric=must_var['metric'],n_jobs=must_var['n_jobs'])
            model.fit(x_train,y_train)
            train_pred=self.my_predict(model,x_train)
            test_pred=self.my_predict(model,x_test)
        self.mymodel=model
        return model,x_train,y_train,train_pred[:,1],x_test,y_test,test_pred[:,1]
    def my_classifier_box(self,box='lgb',must_var={}):
        if box=='logistic':
            model,x_train,y_train,train_pred,x_test,y_test,test_pred=self.my_logis_classifier(must_var=must_var)
            return model,x_train,y_train,train_pred,x_test,y_test,test_pred
        elif box=='lgb':
            model,x_train,y_train,train_pred,x_test,y_test,test_pred=self.my_lgb_classifier(must_var=must_var)
            return model,x_train,y_train,train_pred,x_test,y_test,test_pred
        elif box=='logis':
            model,x_train,y_train,train_pred,x_test,y_test,test_pred=self.my_logis_classifier(must_var=must_var)
            return model,x_train,y_train,train_pred,x_test,y_test,test_pred
        elif box=='forest':
            model,x_train,y_train,train_pred,x_test,y_test,test_pred=self.my_forest_classifier(must_var=must_var)
            return model,x_train,y_train,train_pred,x_test,y_test,test_pred
        elif box=='svm':
            model,x_train,y_train,train_pred,x_test,y_test,test_pred=self.my_svm_classifier(must_var=must_var)
            return model,x_train,y_train,train_pred,x_test,y_test,test_pred
        elif box=='knn':
            model,x_train,y_train,train_pred,x_test,y_test,test_pred=self.my_knn_classifier(must_var=must_var)
            return model,x_train,y_train,train_pred,x_test,y_test,test_pred
