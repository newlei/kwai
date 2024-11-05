import numpy as np 
import pdb 
import pandas as pd


data_path='../'
data_file = pd.read_csv(data_path, sep='|')


for index, row in data_file.iterrows():
    print(index) # 输出每行的索引值

data_good_path='../goods.csv'
pd.read_csv(data_good_path, usecols=['goods_id', 'goods_name','goods_cate_1st_name','goods_cate_2nd_name','goods_cate_3rd_name','brand_name']) # 按列名，列名必须存在  

