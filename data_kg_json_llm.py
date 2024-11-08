import json
import numpy as np 
import pandas as pd 
import pdb



file_name = '../data_process/core10/data_interaction_final.csv'
# data_interaction = pd.read_csv(file_name, usecols=['user_id','photo_id','poi_id','time_second'], sep='|')
data_interaction = pd.read_csv(file_name, usecols=['user_id','poi_id'], sep='|')

file_name = '../data_process/core10/data_interaction_final_cat_u_att.csv'
data_interaction_u_att = pd.read_csv(file_name, sep='|')

file_name = '../data_process/core10/data_interaction_final_cat_poi_att.csv'
data_interaction_poi_att = pd.read_csv(file_name, sep='|')

# file_name = '../data_process/core10/data_interaction_final_cat_p_att.csv'
# data_interaction_photo_att = pd.read_csv(file_name, sep='|')

u_att_dict = dict()
#去重复的
data_interaction_u_att = data_interaction_u_att.drop_duplicates(subset='user_id') 
#u_gender|u_age|u_age_part|u_city|play_duration|u_region
for index, row in data_interaction_u_att.iterrows():
    # print(index) # 输出每行的索引值
    user_id = row['user_id']
    if user_id not in u_att_dict:
        u_att_dict[user_id] = "性别是"+str(row['u_gender'])+",年龄是"+str(row['u_age'])+",居住地是"+str(row['u_city'])+",居住地属于中国"+str(row['u_region'])


poi_att_dict = dict()
#去重复的
data_interaction_poi_att = data_interaction_poi_att.drop_duplicates(subset='poi_id') 
#|user_id|photo_id|time_second|poi_id|poi_name|category_name|cate_2_name|cate_1_name|province_name|city_name|brand_name
for index, row in data_interaction_poi_att.iterrows():
    # print(index) # 输出每行的索引值
    poi_id = row['poi_id']
    if poi_id not in poi_att_dict:
        poi_att_dict[poi_id] = "名称是"+str(row['poi_name'])+",类型是"+str(row['category_name'])+"-"+str(row['cate_2_name'])+"-"+str(row['cate_1_name'])+",所在地是"+str(row['city_name'])


data = []
instruction = "针对时空场景的推荐问题，请总结出用户在时空场景的推荐偏好包括：时间偏好，空间偏好，时空整体偏好，产品类型偏好，总体偏好。每个偏好用一句话描述。" 
data_interaction = data_interaction.groupby('user_id').agg(list).reset_index()
for index, row in data_interaction.iterrows():
    user_id = row['user_id']
    text = "用户"+"ID是："+str(user_id)+","+u_att_dict[user_id]
    text += "\\n 用户交互的产品序列如下：\\n"
    poi_list = row['poi_id']
    for poi_id in poi_list:  
        text = text+ "产品ID是："+str(poi_id)+","+poi_att_dict[poi_id] 
        text+='\\n'
    data.append({
        "instruction": instruction,
        "input": text
    })
    pdb.set_trace()





data.append({
        "instruction": instruction,
        "input": text
    })

output_file = '../data_process/core'+str(10)+'/data_kg_llm.json'

with open(output_file, 'w', encoding='utf-8') as f:
    for item in data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')