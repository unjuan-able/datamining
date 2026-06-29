import pandas as pd
import numpy as np
import toad
import seaborn as sns
import matplotlib.pyplot as plt
from toad.plot import bin_plot
from toad.plot import badrate_plot

class AiFeatureMiner(object):
    def __init__(self, data_set):
        self.my_dataframe = data_set
        self.mybox = None
        self.box_dataframe = None
        self.selected_dataframe = None

    def my_feature_box(self, to_drop, target, box_method, min_sample, num_bins, empty_single, is_bins):
        c = toad.transform.Combiner()
        # 保护机制：如果还没运行过 feature_select，就默认使用原始 dataframe
        base_df = self.selected_dataframe if self.selected_dataframe is not None else self.my_dataframe
        
        c.fit(base_df.drop(to_drop, axis=1, errors='ignore'), y=target, method=box_method, min_sample=min_sample, n_bins=num_bins, empty_separate=empty_single)
        new_df = c.transform(base_df, labels=is_bins)
        self.mybox = c
        self.box_dataframe = new_df
        return new_df
    
    def my_box_detail(self, var):
        box_detail = self.mybox.export()[var]
        return box_detail
    
    def my_box_ineer_plot(self, var, y_target):
        bin_plot(self.box_dataframe, x=var, target=y_target)
    
    def my_box_outer_plot(self, var, y_target, time_var):
        badrate_plot(self.box_dataframe, target=y_target, x=time_var, by=var)

    def my_box_update(self, rule, is_bins):
        self.mybox.update(rule)
        base_df = self.selected_dataframe if self.selected_dataframe is not None else self.my_dataframe
        newdf = self.mybox.transform(base_df, labels=is_bins)
        self.box_dataframe = newdf
        return newdf

    def my_feature_select(self, to_drop, y_target, empty_p, iv_p, corr_p, must_var):
        # 第一次筛选使用原始数据，否则基于上一次筛选结果继续
        base_df = self.selected_dataframe if self.selected_dataframe is not None else self.my_dataframe
        
        # 过滤掉不存在的列，防止 to_drop 报错
        valid_to_drop = [col for col in to_drop if col in base_df.columns]
        
        new_df = toad.selection.select(
            base_df.drop(valid_to_drop, axis=1),
            target=y_target,
            empty=empty_p,
            iv=iv_p,
            corr=corr_p,
            return_drop=False,
            exclude=must_var
        )
        
        # 只有当 valid_to_drop 真的有内容时才进行 concat拼接，否则直接返回 new_df
        if valid_to_drop:
            new_df2 = pd.concat([self.my_dataframe[valid_to_drop], new_df], axis=1)
        else:
            new_df2 = new_df
            
        self.selected_dataframe = new_df2
        return new_df2

    def my_corr_map(self, mydataframe, to_drop):
        valid_to_drop = [col for col in to_drop if col in mydataframe.columns]
        df_corr = mydataframe.drop(valid_to_drop, axis=1).corr()
        corr_map = sns.heatmap(df_corr, annot=False, cmap='PuBuGn')
        return corr_map

    def get_my_dataframe(self):
        return self.my_dataframe

    def get_mybox(self):
        return self.mybox

    def get_box_dataframe(self):
        return self.box_dataframe

    def get_selected_dataframe(self):
        return self.selected_dataframe