from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time
from vllm import LLM, SamplingParams

# Step 1: 初始化模型
# model_path ="Qwen/Qwen2.5-1.5B-Instruct"  
model_path ="Qwen/Qwen2.5-3B-Instruct"
llm = LLM(model=model_path, dtype='half')

# Step 2: 定义批量输入数据
batch_data = [
    "请简述量子计算的基本原理。",
    "给出关于人工智能伦理的三条建议。",
    "如何评价最近的大模型技术发展？",
]

# Step 3: 设置采样参数
sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    max_tokens=1500,  # 限制生成的最大长度
)

# # Step 4: 执行批量推理
# results = llm.generate(batch_data, sampling_params)

# # Step 5: 输出结果
# for i, output in enumerate(results):
#     print(f"输入: {batch_data[i]}")
#     print(f"生成结果: {output.outputs[0].text}\n")

 

count=0
input_texts =[] 
json_path = '../data_process/core'+str(10)+'/data_kg_llm.json'
elapsed_time_all = 0
elapsed_time_count = 0

res_data = []
with open(json_path, 'r', encoding="utf-8") as f:
    # 读取所有行 每行会是一个字符串
    batch_size=0
    batch_data=[]
    batch_data_id = []
    for one_data in f.readlines(): 
        # 将josn字符串转化为dict字典
        start_time = time.time()
        prompt_one = json.loads(one_data)  
        str_in = prompt_one["data"]["instruction"]+"上下文信息："+prompt_one["data"]["input"]+"\n \n 写出总结性的回答，不包含原句重复。"
        batch_data.append(str_in)
        
        # batch_data.append(str(prompt_one["data"])+"\n 请用中文回答")
        batch_data_id.append(prompt_one["user_id"])
        if batch_size<4:
            batch_size+=1
            continue

        response = llm.generate(batch_data, sampling_params)

        # prompt_one = json.loads(one_data)
        # llm_summary(prompt_one)
        # Step 5: 输出结果
        for i, output in enumerate(response):
            print(f"输入 useri id: {batch_data_id[i]}")
            print(f"生成结果: {output.outputs[0].text.strip()}\n")
            res_data.append({
                "user_id":  batch_data_id[i],
                "data": output.outputs[0].text
            }) 

        elapsed_time = time.time() - start_time
        elapsed_time_all+=elapsed_time
        elapsed_time_count+=1
        elapsed_time_average = elapsed_time_all/elapsed_time_count
        print('--each pair time--',elapsed_time,'---avg time--',elapsed_time_average)
        pdb.set_trace()
        batch_data = []
        batch_data_id = []
        

        # if count>4:
        #     break
        # pdb.set_trace()
        # count+=1


json_res_path = '../data_process/core'+str(10)+'/data_kg_llm_summary1.json'
with open(output_file, 'w', encoding='utf-8') as f:
    for item in res_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'


# Each query must come with a one-sentence instruction that describes the task
# task = 'Given a web search query, retrieve relevant passages that answer the query'
# queries = [
#     get_detailed_instruct(task, 'how much protein should a female eat'),
#     get_detailed_instruct(task, 'summit define')
# ]
# # No need to add instruction for retrieval documents
# documents = [
#     "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
#     "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments."
# ]
# input_texts = queries + documents

tokenizer = AutoTokenizer.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)
model = AutoModel.from_pretrained('Alibaba-NLP/gte-Qwen2-7B-instruct', trust_remote_code=True)

max_length = 8192

# # Tokenize the input texts
batch_dict = tokenizer(input_texts, max_length=max_length, padding=True, truncation=True, return_tensors='pt')
# batch_dict = tokenizer([response,response,response,response], max_length=max_length, padding=True, truncation=True, return_tensors='pt')
outputs = model(**batch_dict)
embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

# normalize embeddings
embeddings = F.normalize(embeddings, p=2, dim=1)
scores = (embeddings[:2] @ embeddings[2:].T) #计算向量的相似度。0-1
print(scores.tolist())

print(embeddings)#torch.Size([1, 3584])

pdb.set_trace()
 


 