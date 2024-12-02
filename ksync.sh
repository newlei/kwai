#!/bin/sh
unset -v http_proxy https_proxy no_proxy
export http_proxy=http://10.66.14.33:11080 https_proxy=http://10.66.14.33:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
export http_proxy=http://oversea-squid1.jp.txyun:11080 https_proxy=http://oversea-squid1.jp.txyun:11080 no_proxy=localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com
git status  
git add *  
git commit -m 'add some code from Mac'
# git commit -m 'add some results from Server'
git pull --rebase origin main   #domnload data
git push origin main --force          #upload data
git stash pop
# sleep 3s #Just for more stable, you can remove this.
unset -v http_proxy https_proxy no_proxy