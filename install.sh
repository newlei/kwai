#!/bin/sh
unset -v http_proxy https_proxy no_proxy
export http_proxy=http://10.66.14.33:11080 https_proxy=http://10.66.14.33:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
conda create -n llmrec python=3.10
pip install pandas
pip install torch
pip install transformers
pip install accelerate
pip install optimum
pip install auto-gptq
pip install geopy
pip install joblib
pip install scipy
pip install vllm==0.6.2
pip install gpustat
unset -v http_proxy https_proxy no_proxy