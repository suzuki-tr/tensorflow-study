#!/usr/bin/env python
# -*- coding: utf-8 -*-

# this script downloads images using bing image search

import sys
import os
import re #正規表現操作
import commands as cmd


# クエリ検索したHTMLの取得
def get_HTML(query):
    html = cmd.getstatusoutput("wget -O - https://www.bing.com/images/search?q=" + query)
    return html

# jpg画像のURLを抽出
def extract_URL(html):
    url = []
    sentences = html[1].split('\n')
    ptn = re.compile('<a class="thumb" target="_blank" href="(.+\.jpg)"')

    for sent in sentences:
        print (sent)
        print ('\n')

        if sent.find('<div class="item">') >= 0:
            print (sent)
            print ('\n')

            element = sent.split('<div class="item">')
            for j in range(len(element)):
                mtch = re.match(ptn,element[j])
                if  mtch >= 0:
                    url.append(mtch.group(1))
    return url

# ローカルに画像を保存
def get_IMG(dir,urls):
    for u in urls:
        try:
            os.system("wget -P " + dir + " " + u)
        except:
            continue

if __name__ == "__main__":

    # search key word    
    query = "apple"

    # get search result
    html = get_HTML(query)

    # get image urls
    urls = extract_URL(html)

    for url in urls:
        print url

    # 画像をdirectoryに保存
    get_IMG('images',urls)

