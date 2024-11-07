import numpy as np 
import pdb 
import pandas as pd
import os.path
import sys
import csv
csv.field_size_limit(sys.maxsize)


#交互数据提取：user_id|photo_id|time_second|poi_id
file_name = '../data_process/core10/data_interaction_final.csv'
data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
print(data_interaction.size)




file_user = '../user_pdate_20241104.csv'
user_att = pd.read_csv(file_user, usecols=['user_id','photo_id','time_second','poi_id','label','play_duration','poi_page_stay_time'], sep='|')
user_att.rename(columns={'user_id': 'user_id', 'photo_id': 'gender','time_second': 'age','poi_id': 'age_part','label': 'city','play_duration': 'region'}, inplace=True)
print('user_att',user_att.size)

# file_photo = '../photo_pdate_20241104.csv'
# photo_att = pd.read_csv(file_photo, usecols=['photo_id','poi_id','poi_name','poi_city_name','photo_type','city_name','photo_cate_type','photo_second_cate_type'], sep='|', lineterminator='\n')

file_poi = '../poi_pdate_20241104.csv'
poi_att = pd.read_csv(file_poi, sep='|')
print('poi_att',poi_att.size)

pdb.set_trace()



chunksize = 10 ** 6
file_photo = '../photo_pdate_20241104.csv'
photo_list =[]
for chunk in pd.read_csv(file_photo,  usecols=['photo_id','poi_id','poi_name','poi_city_name','photo_type','city_name','photo_cate_type','photo_second_cate_type'], chunksize=chunksize, sep='|', lineterminator='\n'):
    photo_list.append(chunk) 

print(len(photo_list))

