import numpy as np 
import pdb 
import pandas as pd


data_path='../'
data_file = pd.read_csv(data_path, sep='|')

# good。
# goods_id|seller_id|goods_name|seller_name|goods_cate_1st_name|goods_cate_2nd_name|goods_cate_3rd_name|max_purchase_price_in_yuan|min_purchase_price_in_yuan|item_contents_detail|brand_name|on_sale_status|goods_system_type|part_id
# 100007294130588|2647437896|飞行宝贝飞机票|飞行宝贝欢乐小镇|休闲娱乐|休闲活动|主题乐园|358|358|[{"title":"飞机票","setMealContents":[{"title":"飞机票","count":1,"price":45800}],"fromNum":1,"selectNum":1}]|UNKNOWN|0|1

for index, row in data_file.iterrows():
    print(index) # 输出每行的索引值



