#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         BackTrackExercise
# Author:       wendi
# Date:         2021/11/7

from typing import List


class GenerateParenthesis(object):
    """
    https://leetcode-cn.com/problems/generate-parentheses/
    22. 括号生成
    """

    def solution1(self, n: int) -> List[str]:
        """
        深度优先搜索，减法
        :param n:
        :return:
        """
        res = []
        cur_str = ''

        def dfs(cur_str, left, right):
            """

            :param cur_str: 从根节点到叶子节点的路径字符串
            :param left: 左括号还可以使用的个数
            :param right: 右括号还可以使用的个数
            :return:
            """
            if left == 0 and right == 0:
                res.append(cur_str)
                return
            if right < left:
                return
            if left > 0:
                dfs(cur_str + '(', left - 1, right)
            if right > 0:
                dfs(cur_str + ')', left, right - 1)

        dfs(cur_str, n, n)
        return res

    def solution2(self, n: int) -> List[str]:
        """
        深度优先搜索，加法
        :param n:
        :return:
        """
        res = []
        cur_str = ''

        def dfs(cur_str, left, right, n):
            if left == n and right == n:
                res.append(cur_str)
                return
            if left < right:
                return
            if left < n:
                dfs(cur_str + '(', left + 1, right, n)
            if right < n:
                dfs(cur_str + ')', left, right + 1, n)

        dfs(cur_str, 0, 0, n)
        return res


if __name__ == '__main__':
    obj = GenerateParenthesis()
    res = obj.solution2(3)
    print(res)
