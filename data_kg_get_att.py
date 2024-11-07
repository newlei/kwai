import numpy as np 
import pdb 
import pandas as pd
import os.path

#交互数据提取：user_id|photo_id|time_second|poi_id
file_name = '../data_process/core10/data_interaction_final.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
print(data_interaction.size)


file_user = '../user_pdate_20241104.csv'
user_att = pd.read_csv(file_user, usecols=['user_id','photo_id','time_second','poi_id','label','play_duration','poi_page_stay_time'], sep='|')
user_att.rename(columns={'user_id': 'user_id', 'photo_id': 'gender','time_second': 'age','poi_id': 'age_part','label': 'city','play_duration': 'region'}, inplace=True)


file_photo = '../photo_pdate_20241104.csv'
photo_att = pd.read_csv(file_photo, sep='|', on_bad_lines='skip',lineterminator='\n')

file_poi = '../goods_pdate_20241104.csv'
poi_att = pd.read_csv(file_poi, sep='|')
user_att.rename(columns={'goods_id': 'poi_id'}, inplace=True)


pdb.set_trace()

