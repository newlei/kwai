from transformers import AutoModelForCausalLM, AutoTokenizer
import pdb
import json
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


# model_name = "Qwen/Qwen2.5-7B-Instruct"
model_name ="Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"
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
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    # messages = [prompt]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print(model.device,"model.device")
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device) #"cuda:1")#
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

    print(response)
    return response

# llm_summary(prompt)
# pdb.set_trace()

count=0
input_texts =[] 
json_path = '../data_process/core'+str(10)+'/data_kg_llm.json'

json_res_path = '../data_process/core'+str(10)+'/data_kg_llm_summary.json'

with open(json_path, 'r', encoding="utf-8") as f:
    # 读取所有行 每行会是一个字符串
    for one_data in f.readlines(): 
        # 将josn字符串转化为dict字典
        response_one = llm_summary(one_data) 
        input_texts.append(response_one)
        # prompt_one = json.loads(one_data)
        # llm_summary(prompt_one)
        
        if count>4:
            break
        pdb.set_trace()
        count+=1




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
 



exit()

#目前的显卡是t4的。计算能力弱了，所以支持不了。
# ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0. Your Tesla T4 GPU has compute capability 7.5. You can use float16 instead by explicitly setting the`dtype` flag in CLI, for example: --dtype=half.
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from huggingface_hub import snapshot_download
from vllm.lora.request import LoRARequest
import pdb

#  CUDA_VISIBLE_DEVICES=4  python llm_summary.py
# Initialize the tokenizer
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# max_tokens is for the maximum length for generation.
sampling_params = SamplingParams(temperature=0.7, top_p=0.8,top_k=20, repetition_penalty=1.1, max_tokens=512)
# Input the model name or path. Can be GPTQ or AWQ models.
llm = LLM(model=model_name, enforce_eager=True)

def llm_gen(prompt):
    messages = [
        # {"role": "system", "content": "You are a helpful assistant."},
        {"role": "system", "content": prompt},
    ]

    # text = tokenizer.apply_chat_template(messages,tokenize=False, add_generation_prompt=True)
    # print(text)
    # pdb.set_trace()


    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # generate outputs
    # outputs = llm.generate([text], sampling_params,lora_request=LoRARequest("sql_adapter", 1, lora_local_path=sql_lora_path))
    outputs = llm.generate([text], sampling_params)#,lora_request=LoRARequest("sql_adapter", 1, lora_local_path=sql_lora_path))

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


count=0
json_path = '../data_process/core'+str(10)+'/data_kg_llm.json'
with open(json_path, 'r', encoding="utf-8") as f:
    # 读取所有行 每行会是一个字符串
    for one_data in f.readlines(): 
        # 将josn字符串转化为dict字典
        llm_gen(one_data) 
        # prompt_one = json.loads(one_data)
        # llm_summary(prompt_one)
        if count>4:
            pdb.set_trace()
        count+=1

exit()
