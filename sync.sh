#!/bin/sh
git status  
git add *  
git commit -m 'add some code from Mac'
# git commit -m 'add some results from Server'
git pull --rebase main   #domnload data
git push main --force          #upload data
git stash pop
# sleep 3s #Just for more stable, you can remove this.