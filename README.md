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
代码的最后部分是对id进行reid。

最后生成的数据在

- 拼接了user的属性，少了3K个user，交互从300w到了100w
../data_process/core'+str(10)+'/data_interaction_final_cat_u_att.csv

- 进行reid后的结果
../data_process/core'+str(10)+'/data_interaction_final_cat_u_att_reid.csv

- 拼接了poi的属性，几乎poi都在，交互还是300万，少了一点零头
../data_process/core'+str(10)+'/data_interaction_final_cat_poi_att.csv

- 进行reid后的结果
../data_process/core'+str(10)+'/data_interaction_final_cat_poi_att_reid.csv

- 拼接了photo的属性
../data_process/core'+str(10)+'/data_interaction_final_cat_p_att.csv

- 进行reid后的结果
../data_process/core'+str(10)+'/data_interaction_final_cat_p_att_redi.csv


###  data_kg_json_llm.py
将生成的数据构建成LLM读取的json文件。文件路径

'../data_process/core'+str(10)+'/data_kg_llm.json'

- 还缺一个针对item的数据构建。

###  llm_summary.py
总结用户的时空偏好并提取emb，目前是一条条的送进去，还没有形成batch，因为显存不够。


###  data_pos_behavoir.py
用于找到正样本，便于利用对比学习，将信号对齐。
目前1/(len(i_ulist[i]&i_ulist[j])+alpah)计算过程特别慢，因为要计算所有的i，j的pair情况
- 使用稀疏矩阵来计算就很快了。15s就行了
- user set 和 user set之间的交集，也是先用稀疏矩阵来计算，获得有交集的u v id的pair，然后去计算，常规方法计算预计22 min。采用多线程的方法。550s=9min，少了一点时间，还可以接受。


###  adapter_network.py
adapter网络，因为现在还没有真实数据，给了一下随机的数据进行测试。目前loss稳定下降。
本来只有一个net，现在又搞了一个decoder，用于loss_reconstruction，就成为了一个Unet，感觉不这样的话，只做对齐协同信号，但是维度降低很多，也需要reconstruction来保留语义信息。

