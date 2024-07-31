第一次初始化时修改：$ git init -b main
初始化完成后修改：git branch -m master main
全局方式修改默认分支(暂时还没试过)
git config --global init.defaultBranch main

$ git add .

$ git commit -m "first commit"

$ git branch -M main

$ git remote add origin git@github.com:TTZYTttzyt/A6000.git

第一次提交加-u，后面就不需要了
$ git push -u origin main

把在github上面改的代码直接更新到自己原仓库
git pull


local修改之后如何push?
git status :可查看已经修改的文件
git add xxx或者git add .
git commit -m "xxx"
git push origin main
