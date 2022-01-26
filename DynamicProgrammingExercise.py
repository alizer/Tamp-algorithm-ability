#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         DynamicProgrammingExercise
# Author:       wendi
# Date:         2021/12/15
from typing import List


class UniquePaths:
    """
    https://leetcode-cn.com/problems/unique-paths/ 不同路径
    """
    @staticmethod
    def solution(m: int, n: int) -> int:
        """

        :param m: m行
        :param n: n列
        :return: 总共路径数
        """
        cur = [1] * n
        for i in range(1, m):
            for j in range(1, n):
                cur[j] += cur[j-1]
        return cur[-1]


class UniquePaths2:
    """
    https://leetcode-cn.com/problems/unique-paths-ii/ 不同路径2  带障碍物
    """
    def solution(self, obstacleGrid: List[List[int]]) -> int:
        """
        :param obstacleGrid:
        :return: 不同的路径数
        """
        if not obstacleGrid:
            return 0
        n = len(obstacleGrid)
        m = len(obstacleGrid[0])
        # dp = [[0] * m] * n  这种创建方式有问题，这种是浅拷贝，修改子list中的值，其他的同步会修改
        dp = [[0] * m for _ in range(n)]

        for i in range(m):
            if obstacleGrid[0][i] == 0:
                dp[0][i] = 1
            else:
                break

        for i in range(n):
            if obstacleGrid[i][0] == 0:
                dp[i][0] = 1
            else:
                break

        for i in range(1, n):
            for j in range(1, m):
                if obstacleGrid[i][j] == 1:
                    dp[i][j] = 0
                    continue
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1]
        return dp[-1][-1]


class MinPathSum:
    """
    https://leetcode-cn.com/problems/minimum-path-sum/ 最小路径和
    """
    def solution(self, grid: List[List[int]]) -> int:
        if not grid:
            return 0
        n = len(grid)
        m = len(grid[0])

        for i in range(0, n):
            for j in range(0, m):
                if i == 0 and j == 0:
                    continue
                elif i == 0:
                    grid[i][j] = grid[i][j - 1] + grid[i][j]
                elif j == 0:
                    grid[i][j] = grid[i - 1][j] + grid[i][j]
                else:
                    grid[i][j] = min(grid[i - 1][j], grid[i][j - 1]) + grid[i][j]
        return grid[-1][-1]


class CountBits:
    """
    https://leetcode-cn.com/problems/w3tCBm/
    剑指 Offer II 003. 前 n 个数字二进制中 1 的个数

    输入: n = 5
    输出: [0,1,1,2,1,2]
    解释:
    0 --> 0
    1 --> 1
    2 --> 10
    3 --> 11
    4 --> 100
    5 --> 101

    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/w3tCBm
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """

    def solution1(self, n: int) -> List[int]:
        """
        Brian Kernighan 算法
        :param n:
        :return:
        """
        def countones(x: int) -> int:
            ones = 0
            while x > 0:
                x &= (x - 1)
                ones += 1
            return ones
        bits = [countones(i) for i in range(n+1)]

        return bits

    def solution2(self, n: int) -> List[int]:
        bits = [0]
        highBit = 0
        for i in range(1, n+1):
            if i & (i - 1) == 0:
                highBit = i
            bits.append(bits[i - highBit] + 1)
        return bits


class SingleNumber:
    """
    https://leetcode-cn.com/problems/WGki4K/
    剑指 Offer II 004. 只出现一次的数字

    """
    def solution(self, nums: List[int]):
        ans = 0
        for i in range(32):
            total = sum((num >> i) & 1 for num in nums)
            if total % 3:
                # Python 这里对于最高位需要特殊判断
                if i == 31:
                    ans -= (1 << i)
                else:
                    ans |= (1 << i)
        return ans


class MaxProduct:
    def solution(self, words: List[str]) -> int:
        arr = [0]*len(words)
        for idx, word in enumerate(words):
            for c in word:
                # 1 左移的位数和0取逻辑或的结果，保存对应字母位为0时对应字母不存在，1则存在
                arr[idx] = arr[idx] | (1 << (ord(c) - ord('a')))

        res = 0
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                # 对两两字符串的每一位取逻辑与的结果，若结果为1则说明有对应字母位同时为1，即两字符串存在相同字母
                if arr[i] & arr[j] == 0:
                    res = max(res, len(words[i]) * len(words[j]))

        return res


class CountSubstrings:
    """
    https://leetcode-cn.com/problems/a7VOhD/
    剑指 Offer II 020. 回文子字符串的个数
    """
    def solution(self, s: str) -> int:
        n = len(s)
        # dp[i][j] 表示第i个字符到第j个字符能否组成回文串
        dp = [[0] * n for _ in range(n)]

        ans = 0
        for i in range(n):
            dp[i][i] = 1
            ans += 1

        for i in range(n-2, -1, -1):
            for j in range(i+1, n, 1):
                if s[i] == s[j]:
                    if i < j - 1 and dp[i+1][j-1] == 1:
                        dp[i][j] = 1
                        ans += 1
                    elif i == j - 1:
                        dp[i][j] = 1
                        ans += 1
                else:
                    continue
        return ans

# k个有序数组合并
# 实现栈，pop(),push(),max() 在O(1)的时间内完成


if __name__ == '__main__':
    obj = MaxProduct()
    res = obj.solution(words = ["abcw","baz","foo","bar","fxyz","abcdef"])
    print(res)


