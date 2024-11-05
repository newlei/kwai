import numpy as np 
import pdb 
import pandas as pd


# data_path='../'
# data_file = pd.read_csv(data_path, sep='|')


data_poi_path='../poi_pdate_20241104.csv'
poi_data = pd.read_csv(data_poi_path, usecols=['poi_id', 'poi_name','category_id','category_name','cate_2_id','cate_2_name','cate_1_id','cate_1_name'], sep='|') # 按列名，列名必须存在  

poi_id = set()
for index, row in poi_data.iterrows():
    # print(index) # 输出每行的索引值
    poi_id.add(row['poi_id'])
    # pdb.set_trace()

data_photo_path='../photo_pdate_20241104.csv'
photo_data = pd.read_csv(data_photo_path, usecols=['photo_id', 'poi_id','poi_name','origin_poi_id','photo_cate_type','photo_second_cate_type'], sep='|', lineterminator='\n') # 按列名，列名必须存在  
count_all=0
count = 0
for index, row in photo_data.iterrows():
    # print(index) # 输出每行的索引值
    count_all+=1
    if row['poi_id'] in poi_id:
        count+=1

print(count/count_all)

pdb.set_trace()







exit()


for index, row in data_file.iterrows():
    print(index) # 输出每行的索引值

data_good_path='../goods.csv'
pd.read_csv(data_good_path, usecols=['goods_id', 'goods_name','goods_cate_1st_name','goods_cate_2nd_name','goods_cate_3rd_name','brand_name']) # 按列名，列名必须存在  



data_good_path='../interaction.csv'
pd.read_csv(data_good_path, usecols=['goods_id', 'goods_name','goods_cate_1st_name','goods_cate_2nd_name','goods_cate_3rd_name','brand_name']) # 按列名，列名必须存在  
