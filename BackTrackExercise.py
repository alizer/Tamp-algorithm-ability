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


class LetterCombinations(object):
    def solution(self, digits: str) -> List[str]:
        if not digits: return []
        phone = ['abc','def','ghi','jkl','mno','pqrs','tuv','wxyz']
        queue = ['']  # 初始化队列
        for digit in digits:
            for _ in range(len(queue)):
                tmp = queue.pop(0)
                for letter in phone[ord(digit)-50]:# 这里我们不使用 int() 转换字符串，使用ASCII码
                    queue.append(tmp + letter)
        return queue


class CombinationSum(object):
    def solution(self, candidates: List[int], target: int) -> List[List[int]]:

        def backtrack(candidates, begin, size, path, res, target):

            if 0 == target:
                res.append(path)
                return

            for index in range(begin, size):
                residue = target - candidates[index]
                if residue < 0:
                    break

                backtrack(candidates, index, size, path + [candidates[index]], res, residue)

        size = len(candidates)
        if size == 0:
            return []

        candidates.sort()
        path = []
        res = []
        backtrack(candidates, 0, size, path, res, target)
        return res


class CombinationSum2:
    def solution(self, candidates: List[int], target: int) -> List[List[int]]:

        def dfs(begin, path, residue):
            if residue == 0:
                res.append(path[:])
                return

            for index in range(begin, cnt):
                if residue < candidates[index]:
                    break

                if index > begin and candidates[index - 1] == candidates[index]:
                    continue
                path.append(candidates[index])
                dfs(index + 1, path, residue - candidates[index])
                path.pop()

        cnt = len(candidates)
        if cnt == 0:
            return []
        candidates.sort()
        res = []

        dfs(0, [], target)
        return res


if __name__ == '__main__':
    obj = CombinationSum2()
    res = obj.solution([2,1,2,2,5], 5)
    print(res)
