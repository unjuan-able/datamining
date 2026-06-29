import numpy as np
import pandas as pd
import toad
import hashlib
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
import category_encoders as ce
import matplotlib.pyplot as plt
import seaborn as sns
import copy

class AiDataProcess(object):
    def __init__(self,data_set):
        self.my_dataframe=data_set

    def _hash_encode(self,value):
        hash_object=hashlib.sha256(value.encode())
        hex_dig=hash_object.hexdigest()
        int_dig=int(hex_dig,16)
        return int_dig%10000000
    def my_hash_encode(self,encode_column):
        for col in encode_column:
            self.my_dataframe[col]=self.my_dataframe[col].astype(str).apply(lambda x:self._hash_encode(x))
    def my_eda_count(self):
        my_detect=toad.detect(self.my_dataframe)
        return my_detect
    def my_simple_select(self,to_drop,target,only_iv=True):
        my_quality=toad.quality(self.my_dataframe.drop(to_drop,axis=1),target,iv_only=only_iv)
        return my_quality
    def my_label_encoder(self,col_to_encode):
        label_encoder=LabelEncoder()
        for i in col_to_encode:
            self.my_dataframe[i]=label_encoder.fit_transform(self.my_dataframe[i])
    def my_serial_encoder(self,target_col,org_map):
        self.my_dataframe[target_col]=self.my_dataframe[target_col].map(org_map)
    def my_drop_dup(self,target_col):
        df_new=self.my_dataframe.drop_duplicates(subset=target_col)
        return df_new
    def my_onehot_encoder(self,hot_code=[]):
        df_hot=pd.get_dummies(self,self.my_dataframe,prefix_sep="_",columns=hot_code)
        self.my_dataframe=df_hot
    def my_freq_encoder(self,freq_col):
        ce_encoder=ce.CountEncoder()
        for i in freq_col:
            self.my_dataframe[i]=ce_encoder.fit_transform(self.my_dataframe[i])
    def my_target_encoder(self,y_col,x_col):
        enc=TargetEncoder(cols=x_col)
        x_train=self.my_dataframe.loc[:,x_col]
        y_train=self.my_dataframe.loc[:,y_col]
        train_numeric_dataset=enc.fit_transform(x_train,y_train)
        name_dict={}
        for i in x_col:
            name_dict[i]=str(i)+"_tar"
        train_numeric_dataset=train_numeric_dataset.rename(columns=name_dict)
        mydata=pd.concat([self.my_dataframe,train_numeric_dataset],axis=1)
        self.my_dataframe=mydata
    def my_fillna(self,fill_col_list,fill_col_value):
        value_len=len(fill_col_value)
        if value_len==1 and len(fill_col_list)>0:
            for i in fill_col_list:
                self.my_dataframe[i]=self.my_dataframe[i].fillna(fill_col_value)
        elif value_len>1 and len(fill_col_list)>0:
            for i in range(value_len):
                self.my_dataframe[fill_col_list[i]]=self.my_dataframe[fill_col_list[i]].fillna(fill_col_value[i])
        else:
            self.my_dataframe=self.my_dataframe.fillna(fill_col_value[0])
    def my_find_yccol(self,findstr):
        my_yc_columns=[column for column in self.my_dataframe.columns if self.my_dataframe[column].astype('str').str.contains(findstr).any()]
        return my_yc_columns
    def my_find_ycindex_col(self,col,findstr):
        my_yc_index=self.my_dataframe[self.my_dataframe[col].str.contains(findstr)].index
        return my_yc_index
    def my_find_ycindex_all(self,findstr):
        my_yc_index=self.my_dataframe[self.my_dataframe.astype('str').apply(lambda x:x.str.contains(findstr)).any(axis=1)].index
        return my_yc_index
    def my_drop_yc(self,index_list):
        newdata=self.my_dataframe.drop(index_list)
        newdata=newdata.reset_index(drop=True)
        self.my_dataframe=newdata
    def str_to_str(self,findstr,tostr):
        newdata=self.my_dataframe.replace(findstr,tostr)
        self.my_dataframe=newdata
    def my_box_plot(self,xvar):
        plt.figure(dpi=100)
        box_plot=sns.boxplot(y=self.my_dataframe[xvar],
                             orient='v',
                             flierprops={'marker':'o',
                                         'markerfacecolor':'red',
                                         'color':'black',},
                                         medianprops={'color':'red',
                                          'linestyle':'--'},
                                          showmeans=True,
                                          meanprops={'marker':'D', 'markerfacecolor':'red'})
        plt.show()
    def get_dataframe(self):
        mydata=self.my_dataframe
        return mydata

    
