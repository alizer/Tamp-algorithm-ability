#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         test
# Author:       wendi
# Date:         2021/10/19
# from geopy.geocoders import Nominatim
import time
import random


# class Solution:
#     def lengthOfLongestSubstring(self, s: str) -> int:
#         # 哈希集合，记录每个字符是否出现过
#         occ = set()
#         n = len(s)
#         # 右指针，初始值为 -1，相当于我们在字符串的左边界的左侧，还没有开始移动
#         rk, ans = -1, 0
#         for i in range(n):
#             if i != 0:
#                 # 左指针向右移动一格，移除一个字符
#                 occ.remove(s[i - 1])
#             while rk + 1 < n and s[rk + 1] not in occ:
#                 # 不断地移动右指针
#                 occ.add(s[rk + 1])
#                 rk += 1
#             # 第 i 到 rk 个字符是一个极长的无重复字符子串
#             ans = max(ans, rk - i + 1)
#         return ans


import sys
# import urllib2
import json

if __name__ == '__main__':
    # obj = Solution()
    # res = obj.lengthOfLongestSubstring('pwwkefew')
    #
    # geolocator = Nominatim(user_agent=random.choice(User_Agent_List), timeout=1000)
    # location = geolocator.reverse(input_str, language='zh-CN,zh;q=0.9,en;q=0.8')
    # time.sleep(1)
    # res = location.address
    # print(res)
    input_str = '103.533,10.641'
    url = 'https://restapi.amap.com/v3/geocode/regeo?output=JSON&language=zh&key=c39f275eb61e7b1499963c74475920d0'
    req_url = url + "&location=" + input_str
    res_data = urllib2.urlopen(urllib2.Request(req_url))

    res_dc = json.loads(res_data.read())
    print(res_dc)


    # print(res)
