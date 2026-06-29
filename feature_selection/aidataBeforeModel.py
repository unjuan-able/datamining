import numpy as np
import pandas as pd
import toad
import hashlib

class BuildDataSet(object):
    def __init__(self,data_set):
        self.mydataframe=data_set
        self.mydatameta=None
    
    def _hash_encoder(self,value):
        hash_object=hashlib.sha256(value.encode())
        hex_dig=hash_object.hexdigest()
        int_dig=int(hex_dig)
        return int_dig % 10000000

    def my_hashcode(self,encode_column):
        for col in encode_column:
            self.mydataframe[col]=self.mydataframe[col].astype('str').apply(lambda x: self._hash_encoder(x))
    
    def auto_built(self, target_col,pramary_col,time_col,is_split=0,split_sample=None):
        my_detect=toad.detect(self.mydataframe)
        my_detect=my_detect.reset_index(drop=False)
        object_list=my_detect.loc[my_detect['type']=='object','index'].tolist()
        drop_list=[]
        drop_list=drop_list+pramary_col+time_col
        str_list=[x for x in object_list if x not in drop_list]
        if len(str_list)>0:
            for i in str_list:
                self.my_dataframe[i]=self.my_datafram[i]=self.my_dataframe[i].fillna('未知')
            self.my_hashcode(self.mydataframe,str_list)
        num_list=my_detect.loc[my_detect['type']!='object','index'].tolist()
        num_list2=[x for x in num_list if x not in drop_list]
        df1=self.my_dataframe.loc[:,num_list2]
        df2=self.my_dataframe.loc[:,str_list]
        df3=self.my_dataframe.loc[:,drop_list]
        self.mydatameta=pd.concat([df1,df2,df3],axis=1)
        mydict={}
        mydict['target_col']=target_col
        mydict['pramary_col']=pramary_col
        mydict['time_col']=time_col
        mydict['string_col']=str_list
        num_list.remove(target_col)
        mydict['num_col']=num_list
        self.mydatameta=mydict
        if is_split==0:
            return self.mydataframe,self.mydatameta
        else:
            df_split=self.mydataframe.sample(frac=split_sample)
            sampled_index=df_split.index
            df_org=self.mydataframe[~self.mydataframe.index.isin(sampled_index)]
            return df_org,self.mydatameta,df_split,self.mydatameta
    def arti_built(self, target_col,pramary_col,time_col,is_split=0,split_sample=None):
        my_detect=toad.detect(self.mydataframe)
        my_detect=my_detect.reset_index(drop=False)
        num_list=my_detect.loc[my_detect['type']=='object','index'].tolist()
        drop_list=[]
        drop_list=drop_list+pramary_col+time_col
        num_list2=[x for x in num_list if x not in drop_list]
        num_list2.remove(target_col)
        mydict={}
        mydict['target_col']=target_col
        mydict['pramary_col']=pramary_col
        mydict['time_col']=time_col
        mydict['string_col']=[]
        mydict['num_col']=num_list2
        self.mydatameta=mydict
        if is_split==0:
            return self.mydataframe,self.mydatameta
        else:
            df_split=self.mydataframe.sample(frac=split_sample)
            sampled_index=df_split.index
            df_org=self.mydataframe[~self.mydataframe.index.isin(sampled_index)]
            return df_org,self.mydatameta,df_split,self.mydatameta
            