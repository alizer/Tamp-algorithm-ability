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


class Permute(object):
    def solution(self, nums: List[int]) -> List[List[int]]:
        res = []
        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])
        backtrack(nums, [])
        return res

    def solution2(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path[:])
                return

            for i in range(size):
                if not used[i]:
                    used[i] = True
                    path.append(nums[i])

                    dfs(nums, size, depth + 1, path, used, res)

                    used[i] = False
                    path.pop()

        size = len(nums)
        if len(nums) == 0:
            return []

        used = [False for _ in range(size)]
        res = []
        dfs(nums, size, 0, [], used, res)
        return res


class Permute2(object):
    def solution(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path[:])
                return

            for i in range(size):
                if used[i] or (i > 0 and nums[i] == nums[i-1] and used[i-1]):
                    continue

                used[i] = True
                path.append(nums[i])

                dfs(nums, size, depth + 1, path, used, res)

                used[i] = False
                path.pop()

        size = len(nums)
        if len(nums) == 0:
            return []
        nums.sort()
        used = [False for _ in range(size)]
        res = []
        dfs(nums, size, 0, [], used, res)
        return res


class Combination:
    def solution(self, n: int, k: int) -> List[List[int]]:
        def backtrack(start, n, k):
            if len(path) == k:
                res.append(path[:])
                return

            for i in range(start, n+1):
                if i <= n - k + len(path) + 1:
                    path.append(i)
                    backtrack(i + 1, n, k)
                    path.pop()

        res = []
        path = []
        backtrack(1, n, k)
        return res


class Subsets:
    def solution(self, nums: List[int]) -> List[List[int]]:
        size = len(nums)
        if size == 0:
            return []

        def dfs(nums, start, path):
            res.append(path[:])

            for i in range(start, len(nums)):
                path.append(nums[i])
                # 因为 nums 不包含重复元素，并且每一个元素只能使用一次
                # 所以下一次搜索从 i + 1 开始
                dfs(nums, i + 1, path)
                path.pop()

        res = []
        dfs(nums, 0, [])
        return res


if __name__ == '__main__':
    obj = Subsets()
    res = obj.solution([1,2,3])
    print(res)
