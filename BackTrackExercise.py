#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         BackTrackExercise
# Author:       wendi
# Date:         2021/11/7

from typing import List, Optional


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
    """
    子数组之和等于目标数
    可以重复使用数值
    """
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
    """
    子数组之和等于目标数
    不重复使用数值
    """
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
    """
    保留重复组合
    """
    def solution(self, nums: List[int]) -> List[List[int]]:
        res = []

        def backtrack(nums, tmp):
            if not nums:
                res.append(tmp)
                return
            for i in range(len(nums)):
                # if i > 0 and nums[i] == nums[i-1]:
                #      continue
                backtrack(nums[:i] + nums[i+1:], tmp + [nums[i]])

        backtrack(nums, [])
        return res

    def solution2(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path[:])
                return

            for i in range(len(nums)):
                if not used[i]:
                    used[i] = True
                    path.append(nums[i])
                    dfs(nums, size, depth + 1, path, used, res)
                    used[i] = False
                    path.pop()

        n = len(nums)
        if n == 0:
            return []

        used = [False for _ in range(n)]
        res = []
        dfs(nums, n, 0, [], used, res)
        return res

    def solution3(self, nums: List[int]) -> List[List[int]]:
        ans = []
        if not nums:
            return ans
        self.process3(nums, 0, ans)
        return ans

    def process3(self, arr, index, res):
        """
        arr[0...index-1]都已做好决定
        index后面的字符都可以来到index位置
        不开辟额外的空间
        :param arr:
        :param index:
        :param res:
        :return:
        """
        n = len(arr)
        if index == n:
            res.append(arr[:])
            return
        else:
            for j in range(index, n):
                arr[index], arr[j] = arr[j], arr[index]
                self.process3(arr, index+1, res)
                arr[index], arr[j] = arr[j], arr[index]


class Permute2(object):
    """
    [1,1,2] 去除重复组合
    [[1, 1, 2], [1, 2, 1], [2, 1, 1]]
    """
    def solution(self, nums: List[int]) -> List[List[int]]:
        def dfs(nums, size, depth, path, used, res):
            if depth == size:
                res.append(path[:])
                return

            for i in range(len(nums)):
                if used[i] or (i > 0 and nums[i] == nums[i-1] and used[i-1]):
                    continue

                used[i] = True
                path.append(nums[i])
                dfs(nums, size, depth + 1, path, used, res)
                used[i] = False
                path.pop()

        n = len(nums)
        if n == 0:
            return []
        nums.sort()
        used = [False for _ in range(n)]
        res = []
        dfs(nums, n, 0, [], used, res)
        return res

    def solution2(self, nums: List[int]) -> List[List[int]]:
        ans = []
        if not nums:
            return ans
        self.process2(nums, 0, ans)
        return ans

    def process2(self, arr, index, res):
        """
        arr[0...index-1]都已做好决定
        index后面的字符都可以来到index位置
        不开辟额外的空间
        :param arr:
        :param index:
        :param res:
        :return:
        """
        n = len(arr)
        isVisit = [False]*10
        if index == n:
            res.append(arr[:])
            return

        for j in range(index, n):
            if not isVisit[arr[j]]:
                isVisit[arr[j]] = True
                arr[index], arr[j] = arr[j], arr[index]
                self.process2(arr, index + 1, res)
                arr[index], arr[j] = arr[j], arr[index]


class Combination:
    """
    组合问题
    """
    def solution(self, arr, k: int) -> List[List[int]]:
        """
        包含重复组合
        :param arr: [1, 1, 2]
        :param k: 2
        :return: [[1, 1], [1, 2], [1, 2]]
        """
        def backtrack(start, arr, k):
            if len(path) == k:
                res.append(path[:])
                return

            for i in range(start, len(arr)):
                # 这里加入剪枝策略
                # 搜索起点的上界 = n - 接下来还要选择的元素个数 + 1
                # 接下来还要选择的元素个数 = k-len(path)
                if i <= len(arr) - (k - len(path)) + 1:
                    path.append(arr[i])
                    backtrack(i + 1, arr, k)
                    path.pop()

        res = []
        path = []
        backtrack(0, arr, k)
        return res

    def solution2(self, arr, k: int) -> List[List[int]]:
        """
        去掉重复组合
        :param arr: [1, 1, 2]
        :param k: 2
        :return: [[1, 1], [1, 2]]
        """
        def backtrack(start, arr, k, path, res):
            if len(path) == k:
                res.append(path[:])
                return
            n = len(arr)
            for i in range(start, n):
                # 这里加入剪枝策略
                # 搜索起点的上界 = n - 接下来还要选择的元素个数 + 1
                # 接下来还要选择的元素个数 = k-len(path)
                if i <= n - (k - len(path)) + 1:
                    if i > start and arr[i] == arr[i-1]:
                        continue
                    path.append(arr[i])
                    backtrack(i+1, arr, k, path, res)
                    path.pop()

        res = []
        arr.sort()
        backtrack(0, arr, k, [], res)
        return res


class Subsets:
    """
    https://leetcode.cn/problems/subsets/
    """
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


class Subsets2:
    """
    https://leetcode.cn/problems/subsets-ii/
    """
    def solution(self, nums: List) -> List[List]:
        size = len(nums)
        if size == 0:
            return []

        def dfs(nums, start, path):

            res.append(path[:])
            for i in range(start, len(nums)):
                # 判断是否有重复元素
                if i > start and nums[i] == nums[i - 1]:
                    continue
                path.append(nums[i])
                # 下一次搜索从 i + 1 开始
                dfs(nums, i + 1, path)
                path.pop()

        nums.sort()
        res = []
        dfs(nums, 0, [])
        return res


class Exist(object):
    # 定义上下左右四个行走方向
    # 四个方向的顺序无关紧要
    directs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def solution(self, board, word):
        """
        :type board: List[List[str]]
        :type word: str
        :rtype: bool
        """
        m = len(board)
        if m == 0:
            return False
        n = len(board[0])
        mark = [[0 for _ in range(n)] for _ in range(m)]

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == word[0]:
                    # 将该元素标记为已使用
                    mark[i][j] = 1
                    if self.backtrack(i, j, mark, board, word[1:]):
                        return True
                    else:
                        # 回溯
                        mark[i][j] = 0
        return False

    def backtrack(self, i, j, mark, board, word):
        if len(word) == 0:
            return True

        for direct in self.directs:
            cur_i = i + direct[0]
            cur_j = j + direct[1]

            if 0 <= cur_i < len(board) and 0 <= cur_j < len(board[0]) \
                    and board[cur_i][cur_j] == word[0]:
                # 如果是已经使用过的元素，忽略
                if mark[cur_i][cur_j] == 1:
                    continue
                # 将该元素标记为已使用
                mark[cur_i][cur_j] = 1
                if self.backtrack(cur_i, cur_j, mark, board, word[1:]):
                    return True
                else:
                    # 回溯
                    mark[cur_i][cur_j] = 0
        return False


class StringCombination(object):
    """
    子串
    """
    def solution(self, s: str) -> List[str]:

        def backtrack(s, tmp):
            if tmp:
                res.append(tmp)
            for i in range(len(s)):
                new_s = s[i+1:]
                backtrack(new_s, tmp+s[i])
        res = []

        backtrack(s, '')

        return res


class SubSequences:
    """
    子序列
    """
    def solution(self, s: str):
        """
        打印字符串子序列，有重复
        :param s:
        :return:
        """
        ans = []
        if not s:
            return ans
        self.process1(s, 0, ans, '')
        return ans

    def process1(self, s, index, res, path):
        """
        来到了s[index]字符， s[0..index-1]已经走过了！之前的决定，都在path上
        现在决定是否要s[index]
        :param s:
        :param index:
        :param res:
        :param path:
        :return:
        """
        if index == len(s):
            res.append(path)
            return
        # 不要s[index]字符
        self.process1(s, index+1, res, path)
        # 要s[index]字符
        self.process1(s, index+1, res, path+s[index])

    def solution2(self, s: str):
        """
        打印字符串子序列，无重复
        来到了s[index]字符， s[0..index-1]已经走过了！之前的决定，都在path上
        现在决定是否要s[index]
        :param s:
        :return:
        """
        ans = set()
        if not s:
            return ans
        self.process2(s, 0, ans, '')
        return ans

    def process2(self, s, index, res, path):
        """

        :param s:
        :param index:
        :param res:
        :param path:
        :return:
        """
        if index == len(s):
            res.add(path)
            return

        # 不要s[index]字符
        self.process2(s, index + 1, res, path)
        # 要s[index]字符
        self.process2(s, index + 1, res, path + s[index])


if __name__ == '__main__':
    obj = Subsets2()
    res = obj.solution([i for i in 'aba'])
    print(res)

    # print(get_all_substrings('abc'))
