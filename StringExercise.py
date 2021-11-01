#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         StringExercise
# Author:       wendi
# Date:         2021/10/30
from typing import List
from TreeExercise import TrieTree


class lcp(object):
    """
    最长公共前缀
    """
    def __init__(self):
        super(lcp, self).__init__()

    def longestCommonPrefix_1(self, strs: List[str]):
        """
        巧用zip函数，取每一个单词的同一个位置的字母，判断是否相同
        :param strs:list[str]
        :return:
        """
        res = ''
        for alphabet in zip(*strs):
            tmp_set = set(alphabet)
            if len(tmp_set) == 1:
                res += alphabet[0]
            else:
                break
        return res

    def longestCommonPrefix_2(self, strs: List[str]):
        """
        取一个单词s，和后面单词逐个比较，看s与每个单词相同的最长前缀是多少
        :param strs:
        :return:
        """
        if not strs:
            return ''

        res = strs[0]
        i = 1
        while i < len(strs):
            while strs[1].find(res) != 0:
                res = res[:len(res)-1]
            i += 1
        return res

    def longestCommonPrefix_3(self, strs: List[str]):
        """
        ***横向扫描***
        时间复杂度：O(mn)，m是字符串的平均长度，n是字符串个数
        空间复杂度：O(1)
        :param strs:
        :return:
        """
        if not strs:
            return ''

        prefix, count = strs[0], len(strs)
        for i in range(1, count):
            prefix = self.lcp(prefix, strs[i])
            if not prefix:
                break
        return prefix

    def lcp(self, str1, str2):
        length, index = min(len(str1), len(str2)), 0
        while index < length and str1[index] == str2[index]:
            index += 1
        return str1[:index]

    def longestCommonPrefix_4(self, strs: List[str]):
        """
        ***纵向比较***
        时间复杂度：O(mn)，m是字符串的平均长度，n是字符串个数
        空间复杂度：O(1)
        :param strs:
        :return:
        """
        if not strs:
            return ''
        length, count = len(strs[0]), len(strs)
        for i in range(length):
            c = strs[0][i]
            if any(i == len(strs[j]) or strs[j][i] != c for j in range(1, count)):
                return strs[0][:i]
        return strs[0]

    def longestCommonPrefix_5(self, strs: List[str]):
        """
        ***分治法***
        时间复杂度：O(mn)，m是字符串的平均长度，n是字符串个数
        空间复杂度：O(mlon(n))
        :param strs:
        :return:
        """
        def lcp(start, end):
            if start == end:
                return strs[start]

            mid = (start + end)//2
            lcpLeft = lcp(start, mid)
            lcpRight = lcp(mid+1, end)
            minLength = min(len(lcpLeft), len(lcpRight))
            for i in range(minLength):
                if lcpLeft[i] != lcpRight[i]:
                    return lcpLeft[:i]

            return lcpLeft[:minLength]

        return '' if not strs else lcp(0, len(strs)-1)

    def longestCommonPrefix_6(self, strs: List[str]):
        """
        ***二分查找***
        时间复杂度：O(mnlog(m))，m是字符串的平均长度，n是字符串个数
        空间复杂度：O(1)
        :param strs:
        :return:
        """
        def isCommonPrefix(length):
            str0, count = strs[0][:length], len(strs)
            return all(strs[i][:length] == str0 for i in range(1, count))

        if not strs:
            return ''

        minLength = min(len(s) for s in strs)
        low, high = 0, minLength
        while low < high:
            mid = (high - low + 1) // 2 + low
            if isCommonPrefix(mid):
                low = mid
            else:
                high = mid - 1

        return strs[0][:low]

    def longestCommonPrefix_7(self, strs: List[str]):
        """
        ***字典树Trie***
        时间复杂度：
        空间复杂度：
        :param strs:
        :return:
        """

        if not strs:
            return ''

        trieTree = TrieTree()
        for str in strs:
            trieTree.insert(str)

        lcp = []
        while trieTree.flag != 1:
            tmp = [node is not None for node in trieTree.children]
            childNode = sum(tmp)
            if childNode == 1:
                curNodeidx = tmp.index(True)
                curNodeChr = chr(ord('a') + curNodeidx)
                lcp.append(curNodeChr)
                trieTree = trieTree.children[curNodeidx]
            else:
                break

        return ''.join(lcp)




if __name__ == '__main__':
    obj = lcp()
    res = obj.longestCommonPrefix_5(['abc', 'af', 'adsf', 'asdfdf'])
    print(res)