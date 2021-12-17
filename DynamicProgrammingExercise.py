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


if __name__ == '__main__':
    obj = MinPathSum()
    res = obj.solution(grid=[[1,2],[1,1]])
    print(res)
