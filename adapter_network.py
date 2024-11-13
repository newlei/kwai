# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 
import torch.nn.functional as F

import argparse
import os
import numpy as np
import math
import sys

os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [1]))

from torch.utils.data import DataLoader
from torchvision import datasets
import random
import torch.utils.data as data
import pdb
import time


class Adapter(nn.Module):
    def __init__(self,neg_sample):
        super(Adapter, self).__init__() 
        
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5) 
        # First fully connected layer
        self.fc1 = nn.Linear(3584, int(3548/4)) #3584-896
        self.fc2 = nn.Linear(int(3584/4), int(3584/16))#896-224
        self.fc3 = nn.Linear(int(3584/16), int(3584/64))#224-56
        self.fc4 = nn.Linear(int(3584/64), int(3584/128))#56-28

        self.net = nn.Sequential(
            nn.Linear(3584, int(3584/4)),#3584-896
            nn.Dropout(0.25),
            nn.Tanh(),
            # nn.LeakyReLU(0.2,inplace=True),
            nn.Linear(int(3584/4), int(3584/16)),#896-224
            nn.Dropout(0.25),
            nn.Tanh(),
            nn.Linear(int(3584/16), int(3584/64)),#224-56
            nn.Dropout(0.25),
            nn.Tanh(),
            nn.Linear(int(3584/64), int(3584/128)),#56-28
            nn.Tanh(),
        )
        self.temperature = 0.7
        self.neg_sample = neg_sample#10
    def forward(self, input_emb, pos_emb, input_copy_emb, neg_emb):
        
        input_emb_f = self.net(input_emb)
        pos_emb_f = self.net(pos_emb)
        input_copy_emb_f = self.net(input_copy_emb)
        neg_emb_f = self.net(neg_emb)

        cos_sim_pos = F.cosine_similarity(input_emb_f, pos_emb_f, dim=-1)/self.temperature
        cos_sim_neg = F.cosine_similarity(input_copy_emb_f, neg_emb_f, dim=-1)/self.temperature
        loss_base = torch.exp(cos_sim_pos)/torch.exp(cos_sim_neg).view(-1, self.neg_sample).sum(dim=1)
        loss = -torch.log(loss_base).mean(-1)
        return loss

    def output_emb(self,input_emb):
        output_emb = self.net(input_emb)
        return output_emb


class embData(data.Dataset):
    def __init__(self,emb_dict=None,neg_sample=0,pair_dict=None):
        super(embData, self).__init__() 
      
        self.emb_dict = emb_dict   
        self.pair_dict = pair_dict #用户pos的列表。
        self.neg_sample = neg_sample
        self.all_id = []
        for id_one in self.pair_dict:
            self.all_id.append(id_one)
        self.all_id_set = set(self.all_id)
    def __len__(self):  
        return len(self.all_id) #math.ceil(len(self.emb_dict)*1.0/self.batch_size)#

    def __getitem__(self, idx): 
        # u=[] u_pos=[] u_copy = [] u_neg = []
        u_id = self.all_id[idx] 
        pos_id_list  = self.pair_dict[u_id]
        pos_id = np.random.choice(pos_id_list,1)[0]#random.sample(set(pos_id_list),1)
        # print(pos_id,'----') 

        u = self.emb_dict[u_id]
        u_pos = self.emb_dict[pos_id]

        u_copy = [u]*self.neg_sample
        neg_id_list = self.all_id_set -set(pos_id_list)
        u_neg_id = np.random.choice(list(neg_id_list),self.neg_sample) #random.sample(neg_id_list,self.neg_sample)
        u_neg = [self.emb_dict[k] for k in u_neg_id]# self.emb_dict[u_neg_id]
 
        #实际上只用到一半去计算，不需要j的。
        return torch.from_numpy(np.array(u)), torch.from_numpy(np.array(u_pos)), torch.from_numpy(np.array(u_copy)), torch.from_numpy(np.array(u_neg))           
 
 

neg_sample =10
batch_size = 128
model = Adapter(neg_sample)
model = model.to('cuda') 

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.005)#, betas=(0.5, 0.99))

# emb_dict  #是个dict，dict[i]=emb，emb是llm得到的。
# pair_dict #是个dict，dict[i]=postive of i，通过data_pos_behavior.py得到的，dict[user]的postive user of dict[user], dict[item]的postive item of dict[item]

# 测试的案例 emb_dict and pair_dict
user_len = 1000
emb_dict =dict()
pair_dict =dict()
all_user_list = np.array(range(user_len))
for i in range(user_len): 
    emb_dict[i] = np.random.rand(3584).astype(np.float32)
    pair_dict[i] = np.random.choice(all_user_list,np.random.randint(98)+2)
    # np.random.choice(list(all_user_list),np.random.randint(98)+2)
    # pdb.set_trace()


# emb_dict=None,neg_sample=0,pair_dict=None
train_dataset = embData(
        emb_dict=emb_dict, neg_sample=neg_sample, pair_dict=pair_dict)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=2)

# testing_dataset_loss = embData(
#         emb_dict=emb_dict, neg_sample=neg_sample, pair_dict=pair_dict_test)
# testing_loader_loss = DataLoader(testing_dataset_loss,
#         batch_size=batch_size, shuffle=False, num_workers=0)


########################### TRAINING #####################################
print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(150):
    model.train() 
    start_time = time.time()
    train_loss_sum=[] 
    for u_emb, pos_emb, u_copy,neg_emb in train_loader:
        u_emb = u_emb.cuda()
        pos_emb = pos_emb.cuda()
        u_copy = u_copy.cuda() 
        neg_emb = neg_emb.cuda()

        model.zero_grad()
        loss= model(u_emb, pos_emb, u_copy, neg_emb) 
        loss.backward()
        optimizer_bpr.step() 
        count += 1  
        train_loss_sum.append(loss.item())     
        # print(count)
    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了 
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)
    print('--train--',str_print_train)

    # PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    # torch.save(model.state_dict(), PATH_model)

model.eval() 
emb_dict_list = []
for i in emb_dict:
    emb_dict_list.append(emb_dict[i])
emb_dict_list_input = torch.from_numpy(np.array(emb_dict_list)).cuda()
emb_dict_learned = model.output_emb(emb_dict_list_input)
pdb.set_trace()
# emb_dict_learned_path = './emb_dict_learned.npy'
# np.save(emb_dict_learned_path,emb_dict_learned)
# np.load(emb_dict_learned_path)



