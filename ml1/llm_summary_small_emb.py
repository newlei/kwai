from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time
from vllm import LLM, SamplingParams
import numpy as np
# CUDA_VISIBLE_DEVICES=4  python llm_summary_small_emb.py


# Create an LLM.
# model = LLM(model="intfloat/e5-mistral-7b-instruct", enforce_eager=True)
model = LLM(model="../../gte-Qwen2-7B-instruct", task="embedding", enforce_eager=True)
# model = LLM(model='Alibaba-NLP/gte-Qwen2-7B-instruct', enforce_eager=True, tensor_parallel_size=2)
# Generate embedding. The output is a list of EmbeddingRequestOutputs.

# json_path = '../../data/data_llm_summary_user_summary.json'
json_path = '../../data/data_llm_summary_item_summary.json'
elapsed_time_all = 0
elapsed_time_count = 0

user_emb_dict = dict()

with open(json_path, 'r', encoding="utf-8") as f:
    batch_size=0
    batch_data=[]
    batch_data_id = []
    for one_data in f.readlines(): 
        # 将josn字符串转化为dict字典
        start_time = time.time()
        prompt_one = json.loads(one_data)  
        # str_in = prompt_one["data"]["instruction"]+"上下文信息："+prompt_one["data"]["input"]+"\n \n 写出总结性的回答，不包含原句重复。"
        # batch_data.append(str_in)

        batch_data.append(str(prompt_one["data"]))
        # batch_data_id.append(prompt_one["user_id"])
        batch_data_id.append(prompt_one["poi_id"])
        if batch_size<=2046:
            batch_size+=1
            continue
        try:
            outputs = model.encode(batch_data)
        except:
            empty_items = [i for i, v in enumerate(batch_data) if not v]
            pdb.set_trace()
            
        # Print the outputs.
        for i in range(len(outputs)):
            user_id  = batch_data_id[i]
            user_emb = outputs[i] 

            if user_id not in user_emb_dict:
                user_emb_dict[user_id] = user_emb.outputs.embedding #3584
            else:
                print('double user id error')
                pdb.set_trace()


            # print(outputs[i].outputs.embedding)  # list of 4096 floats
        # pdb.set_trace()

        elapsed_time = time.time() - start_time
        elapsed_time_all+=elapsed_time
        elapsed_time_count+=1
        elapsed_time_average = elapsed_time_all/elapsed_time_count
        print('--each pair time--',elapsed_time,'---avg time--',elapsed_time_average)
        # pdb.set_trace()
        batch_data = []
        batch_data_id = []
        batch_size = 0

    if batch_size>0:
        outputs = model.encode(batch_data)
        # Print the outputs.
        for i in range(len(outputs)):
            user_id  = batch_data_id[i]
            user_emb = outputs[i] 

            if user_id not in user_emb_dict:
                user_emb_dict[user_id] = user_emb.outputs.embedding #3584
            else:
                print('double user id error')
                pdb.set_trace()


# np.save('../../data/llm_user_emb.npy',user_emb_dict)
np.save('../../data/llm_item_emb.npy',user_emb_dict)

pdb.set_trace()
user_emb_dict1 = np.load('../../data/llm_user_emb.npy')

exit()