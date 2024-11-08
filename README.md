# kwai

### data_kg.py

是用于清洗点击的行为数据，最后生成的数据在

'../data_process/core'+str(10)+'/data_interaction_final.csv

文件中的实例如下：
|user_id|photo_id|time_second|poi_id
0|8686|100014228410658|1729353600|3002790283314925585
1|8686|100014228410658|1729094400|3002790283314925585


### data_kg_get_att.py

是用于获得user_id ,photo_id,|poi_id,对应的属性信息。

最后生成的数据在

拼接了user的属性，少了3K个user，交互从300w到了100w
../data_process/core'+str(10)+'/data_interaction_final_cat_u_att.csv
拼接了poi的属性，几乎poi都在，交互还是300万，少了一点零头
../data_process/core'+str(10)+'/data_interaction_final_cat_poi_att.csv
拼接了photo的属性
../data_process/core'+str(10)+'/data_interaction_final_cat_p_att.csv


### 
将生成的数据构建成LLM读取的json文件。文件路径

'../data_process/core'+str(10)+'/data_kg_llm.json'
