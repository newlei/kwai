from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import time

# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name ="Qwen/Qwen2.5-1.5B-Instruct"#"Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
# from_pretrained(model_path, device_map = "balanced_low_0")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt =  "针对时空场景的推荐问题，用户交互行为如下：点击产品1（肯德基套餐）,时间：周六的晚上，空间：商场；点击产品2（麦当劳套餐）,时间：周一的晚上，空间：公司。对于时间信息先分类成工作日、节假日、上午、中午、晚上这类细粒度的时间信息，对于空间信息区分开1km，3km，5km这类细粒度信息，然后请总结出用户在时空场景的推荐偏好：从时间偏好，空间偏好，时空整体偏好，产品类型偏好。"#"Give me a short introduction to large language model."


# You are a helpful assistant.你是一个推荐系统的助手
def llm_summary(prompt):
    print(model.device,"model.device")

    model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device) #"cuda:1")#
    # model_inputs = tokenizer(prompt, return_tensors="pt").to(model.device) #"cuda:1")#
    # sampling_params = SamplingParams(temperature=0.7, top_p=0.8,top_k=20, repetition_penalty=1.1, max_tokens=1024)
    generated_ids = model.generate(
        **model_inputs,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.1,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    pdb.set_trace()
    
    print(response)
    return response

# llm_summary(prompt)
# pdb.set_trace()

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
    for one_data in f.readlines(): 
        # 将josn字符串转化为dict字典
        start_time = time.time()
        prompt_one = json.loads(one_data)  

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": str(prompt_one["data"])}
        ]
        # messages = [prompt]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        batch_data.append(text)
        if batch_size<2:
            batch_size+=1
            continue
        
        # response_one = llm_summary(str(prompt_one["data"]))
        response_one = llm_summary(batch_data)
        input_texts.append(response_one)

        # prompt_one = json.loads(one_data)
        # llm_summary(prompt_one)
        res_data.append({
            "user_id":  prompt_one["user_id"],
            "data": response_one
        }) 

        elapsed_time = time.time() - start_time
        elapsed_time_all+=elapsed_time
        elapsed_time_count+=1
        elapsed_time_average = elapsed_time_all/elapsed_time_count
        print('--each pair time--',elapsed_time,'---avg time--',elapsed_time_average)
        batch_data = []
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
 


 