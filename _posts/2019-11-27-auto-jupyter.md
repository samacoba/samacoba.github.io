---
layout: post
title: jupyter自動立ち上げメモ
category: blog
tags: Qiita
---

ubuntu18.04で起動時に自動でjupyterを立ち上げれるようにするときの設定

jupyterが立ち上がったとしても、cuda, cupy の環境変数あたりの読み込みがうまくいってなかったようで、スクリプト上で ./bashrc を強制的に読み込ませた

今回の動作ユーザ名は「samacoba」で、適宜書き換えが必要

### jupyter立ち上げ用スクリプトを作成

systemdからだとbash環境が読み込まれていないようで、
bashrcを読み込んだあと、jupyterを実行するスクリプトを作成（とりあえずDesktopに置いた）

* ~/Desktop/run_jupyter.sh

```

#!/bin/bash

source ~/.bashrc
jupyter-notebook

```

### 実行権限を追加

```
$ chmod +x Desktop/run_jupyter.sh 
```

### .bashrcのブロックをコメントアウト

https://qiita.com/takaram/items/17e739e9b7d4d7b6de42

スクリプト上でsource ~/.bashrcをしても、ブロックされるので、ブロックしているところをコメントアウトする

* ~/.bashrc

```
# If not running interactively, don't do anything
#case $- in
#    *i*) ;;
#      *) return;;
#esac
```
### sytemdのserviceファイルの作成

https://qiita.com/k0kubun/items/3c94473506e0e370a227

systemdのサービスファイルを作成する  
systemd/userフォルダがないときは作成

* ~/.config/systemd/user/jupyter.service

```
[Unit]
Description=Jupyter Notebook

[Service]
Type=simple
WorkingDirectory=/home/samacoba
ExecStart=/bin/bash /home/samacoba/Desktop/run_jupyter.sh

[Install]
WantedBy=default.target

```

### sytemdサービス自動起動有効

```
$ systemctl --user enable jupyter

```

⇒再起動すると自動でjupyterが立ち上がっていた


### systemctl　サービスコマンド一覧

https://qiita.com/sinsengumi/items/24d726ec6c761fc75cc9


