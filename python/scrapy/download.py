#!/usr/bin/env python
# coding=utf-8

# -------------------------------------------
# File Name: download.py
# Written by Pierre_Hao
# Mail: hao.zhang@1000look.com
# Created Time: Wed 02 Mar 2016 03:44:28 PM CST
# Copyright (c) 2015 Nanjing Qingsou
# -------------------------------------------

import socket
#set time out
socket.setdefaulttimeout(20.0)
import urllib
import os.path as osp
import os
import time

downloads = urllib.urlretrieve

def main(save_folder='/media/F/dataset/TripletClothes1/', txt="photos.txt"):
    sleepTime = 10.0
    f = open(txt, 'r')
    lines = f.readlines()
    flag = 0
    for line in lines:
        if len(line) > 2:
            string = line.strip().split(',')
            name = string[0]
            url = string[1]
            save_path = osp.join(save_folder,str(int(name))+'.jpg')
            if not osp.exists(save_path):
                while True:
                    try:
                        downloads(url, save_path)
                        break
                    except:
                        print 'Downloading ', url, 'error'
                        print 'Sleep time: ', sleepTime, 's'
                        time.sleep(sleepTime)
                        sleepTime += 5
                        #downloads(url, save_path)
            flag += 1
        if flag % 100 == 0:
            print 'Downloading images: ', flag

if __name__ == '__main__':
    main()

