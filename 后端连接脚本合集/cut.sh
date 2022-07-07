#!/bin/bash
editcap -c 5000 $1 $1    #￥1前可加路径
./up.py
./image_preprocess.py