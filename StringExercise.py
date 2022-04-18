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


class GetMaxRepeatSubLen(object):
    """
        求一个字符串中连续出现最多的子串次数
        https://blog.csdn.net/u012333003/article/details/39230493
    """
    def solution(self, input: str):
        """
        时间复杂度：O(n^3)，n 为字符串长度
        空间复杂度：O(1)
        :param input: abcbcbcabc
        :return: 连续出现次数最多的子串是bc，出现次数为3。
        """

        max_cnt = 1
        input_len = len(input)
        for i in range(input_len):
            for j in range(i+1, input_len//2):
                sub_str = input[i:j]
                offset = j - i
                if sub_str == input[j:j+offset]:
                    count = 2
                    for k in range(j+offset, input_len, offset):
                        if sub_str == input[k:k+offset]:
                            count += 1
                        else:
                            break
                        if count > max_cnt:
                            max_cnt = count

        return max_cnt

class MinWindow(object):
    """
    剑指 Offer II 017. 含有所有字符的最短字符串
    https://leetcode-cn.com/problems/M1oyTv/
    """
    def solution(self, s: str, t: str) -> str:
        import collections
        dc_s = collections.defaultdict(int)
        dc_t = collections.defaultdict(int)
        for c in t:
            dc_t[c] += 1

        lp = 0
        valid_cnt = 0
        ans = ""
        for rp in range(len(s)):
            dc_s[s[rp]] += 1
            if dc_s[s[rp]] <= dc_t[s[rp]]:
                valid_cnt += 1

            while lp < rp and dc_s[s[lp]] > dc_t[s[lp]]:
                dc_s[s[lp]] -= 1
                lp += 1

            if valid_cnt == len(t):
                if not ans or rp-lp+1 < len(ans):
                    ans = s[lp:rp+1]
        return ans


class LongestPalindrome(object):
    """
    5. 最长回文子串
    https://leetcode-cn.com/problems/longest-palindromic-substring/
    """
    def solution(self, s: str) -> str:
        """
        时间复杂度：O(n^2)，两层循环
        空间复杂度：O(1)
        :param s:
        :return:
        """
        start, end = 0, 0
        for i in range(len(s)):
            left1, right1 = self.expandAroundCenter(s, i, i)
            left2, right2 = self.expandAroundCenter(s, i, i+1)

            if right1 - left1 > end - start:
                start, end = left1, right1
            if right2 - left2 > end - start:
                start, end = left2, right2
        return s[start:end+1]

    def expandAroundCenter(self, s, left, right):
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return left+1, right-1

    def solution2(self, s):
        """
        马拉车算法
        Manacher‘s Algorithm
        :param s:
        :return:
        """
        t = self.preProcess(s)
        print(t)
        n = len(t)
        p = [0]*n
        C, R = 0, 0
        for i in range(1, n-1):
            i_mirror = 2 * C - i
            print(f'C, R: {C}, {R}')
            if R > i:
                p[i] = min(R-i, p[i_mirror])
            else:
                p[i] = 0

            while t[i+1+p[i]] == t[i-1-p[i]]:
                p[i] += 1

            if i + p[i] > R:
                C = i
                R = i + p[i]

        max_len = 0
        center_idx = 0
        for i in range(1, n-1):
            if p[i] > max_len:
                max_len = p[i]
                center_idx = i
        start = (center_idx - max_len) // 2

        return s[start:start+max_len]

    def preProcess(self, s):
        if not s:
            return "^$"
        ret = "^"
        for c in s:
            ret += "#" + c
        ret += "#$"

        return ret


class PrintMatrix:
    """
    顺时针打印数组
    """
    def solution(self, matrix):
        rows = len(matrix)
        cols = len(matrix[0])
        result = []
        if rows == 0 and cols == 0:
            return result
        left, right, top, bottom = 0, cols-1, 0, rows-1
        while left <= right and top <= bottom:
            # from left to right
            for i in range(left, right+1):
                result.append(matrix[top][i])
            # from top to bottom
            for i in range(top+1, bottom+1):
                result.append(matrix[i][right])
            # from right to left
            if top != bottom:
                for i in range(left, right)[::-1]:
                    result.append(matrix[bottom][i])
            # from bottom to top
            if left != right:
                for i in range(top+1, bottom)[::-1]:
                    result.append(matrix[i][left])
            left += 1
            right += 1
            bottom += 1
            top += 1

        return result

    def solution2(self, matrix):
        """
        魔方
        首先取走矩阵的第一行，然后逆时针翻转矩阵，把最后一列翻转到第一行。
        接着继续取走第一行，然后再继续翻转矩阵，一直到取走所有元素，就是按顺时针打印出来的。
        :param matrix:
        :return:
        """

        # 该函数像魔方一样翻转矩阵，每次都把要打印的一列翻转到第一行
        def turnMagic(matrix):
            rows = len(matrix)
            cols = len(matrix[0])
            arr = []
            for i in range(cols-1, -1, -1):
                tmp = []
                for j in range(rows):
                    tmp.append(matrix[j][i])
                arr.append(tmp)
            return arr

        res = []
        while matrix:
            res += matrix.pop(0)
            if not matrix:
                break
            matrix = turnMagic(matrix)
        return res


if __name__ == '__main__':
    obj = PrintMatrix()
    res = obj.solution2([[1,2,3],[5,6,7,],[9,10,11]])
    print(res)