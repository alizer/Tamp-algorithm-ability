#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         StackExercise
# Author:       wendi
# Date:         2021/11/4
from typing import List


class ParenthesesValid(object):
    def isValid(self, s: str) -> bool:
        stack = []
        for pare in s:

            if pare in '([{':
                stack.append(pare)

            if not stack or (pare == ')' and stack[-1] != '('):
                return False

            elif not stack or (pare == ']' and stack[-1] != '['):
                return False

            elif not stack or (pare == '}' and stack[-1] != '{'):
                return False

            if pare in ')]}':
                stack.pop()
        return not stack


class MinStack(object):
    """
    实现一个能在O(1)时间复杂度 完成 Push、Pop、Min操作的栈
    空间复杂度 O(n)
    """
    def __init__(self):
        # 存放所有元素
        self.stack = []
        # 存放每一次压入数据时，栈中的最小值
        # （如果压入数据的值大于栈中的最小值就不需要重复压入最小值，
        # 小于或者等于栈中最小值则需要压入）
        self.minStack = []

    def push(self, x):
        self.stack.append(x)
        if not self.minStack or self.minStack[-1] >= x:
            self.minStack.append(x)

    def pop(self):
        # 移除栈顶元素时，判断是否移除栈中最小值
        if self.minStack[-1] == self.stack[-1]:
            del self.minStack[-1]
        self.stack.pop()

    def getMin(self):
        return self.minStack[-1]


class MinStack2(object):
    """
    实现一个能在O(1)时间复杂度 完成 Push、Pop、Min操作的栈
    空间复杂度 O(1)
    """
    def __init__(self):
        # 存放所有元素
        self.stack = []
        self.min_value = None

    def push(self, x):
        if not self.min_value:
            self.min_value = x
        self.stack.append(self.min_value - x)
        if self.stack[-1] > 0:
            self.min_value = x

    def pop(self):
        if self.stack[-1] > 0:
            self.min_value = self.min_value + self.stack[-1]
        self.stack.pop()

    def getMin(self):
        return self.min_value


class AsteroidCollision:
    def solution(self, asteroids):
        s, p = [], 0
        while p < len(asteroids):
            if not s or s[-1] < 0 or asteroids[p] > 0:
                s.append(asteroids[p])
            elif s[-1] <= -asteroids[p]:
                if s.pop() < -asteroids[p]:
                    continue
            p += 1
        return s


class DailyTemperatures:
    def solution(self, temperatures: List[int]):
        n = len(temperatures)
        if n == 1:
            return [0]
        res = [0] * n
        stack = []
        for i in range(n):
            if not stack or temperatures[stack[-1]] >= temperatures[i]:
                stack.append(i)
            elif stack and temperatures[stack[-1]] < temperatures[i]:
                while stack and temperatures[stack[-1]] < temperatures[i]:
                    idx = stack[-1]
                    res[idx] = i - stack.pop()
                stack.append(i)
        return res


class InorderTraversal:
    """
    二叉树
    中序遍历
    """
    def solution(self, root):
        """
        循环实现
        :param root:
        :return:
        """
        stack = []
        res = []
        curr = root

        while stack or curr:
            if curr:
                stack.append(curr)
                curr = curr.left
            else:
                curr = stack.pop()
                res.append(curr.val)
                curr = curr.right

    def solution2(self, root):
        """
        递归实现
        :param root:
        :return:
        """
        if not root:
            return []

        return self.solution2(root.left) + [root.val] + self.solution2(root.right)


class LargestRectangleArea:
    """
    剑指 Offer II 039. 直方图最大矩形面积
    https://leetcode-cn.com/problems/0ynMMM/
    """
    def solution(self, heights: List[int]) -> int:
        """
        单调栈【栈中元素 单调递增】
        :param heights:
        :return:
        """
        stack = [-1]
        max_area = 0
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                height = heights[stack[-1]]
                stack.pop()
                width = i - stack[-1] - 1
                max_area = max(max_area, height * width)
            stack.append(i)

        while stack[-1] != -1:
            height = heights[stack[-1]]
            stack.pop()
            width = len(heights) - stack[-1] - 1
            max_area = max(max_area, height * width)
        return max_area


class MaximalRectangle:
    """
    剑指 Offer II 040. 矩阵中最大的矩形
    https://leetcode-cn.com/problems/PLYXKQ/
    """
    def solution(self, matrix: List[str]):
        """
        单调栈，上一题的应用
        :param matrix:
        :return:
        """

        def largestRectangleArea(heights: List[int]) -> int:
            """
            单调栈【栈中元素 单调递增】
            :param heights:
            :return:
            """
            stack = [-1]
            max_area = 0
            for i in range(len(heights)):
                while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                    height = heights[stack[-1]]
                    stack.pop()
                    width = i - stack[-1] - 1
                    max_area = max(max_area, height * width)
                stack.append(i)

            while stack[-1] != -1:
                height = heights[stack[-1]]
                stack.pop()
                width = len(heights) - stack[-1] - 1
                max_area = max(max_area, height * width)
            return max_area

        if len(matrix) == 0:
            return 0
        max_area = 0
        heights = [0 for _ in range(len(matrix[0]))]
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if matrix[i][j] == '0':
                    heights[j] = 0
                else:
                    heights[j] += 1
            max_area = max(max_area, largestRectangleArea(heights))
        return max_area


class ExpressionCompute:
    """
    计算表达式的值：-2+35*((7+8)*(6-3))
    """
    def solution(self, s):
        return self.calc(s, 0)[0]

    def calc(self, s, i):
        """
        请从str[i...]往下算，遇到字符串终止位置或者右括号，就停止
        返回两个值，长度为2的数组
        0) 负责的这一段的结果是多少
        1) 负责的这一段计算到了哪个位置
        :param s: -2+5*((7+8)*(6-3))
        :param i: index
        :return:
        """
        stack = []
        cur = 0
        while i < len(s) and s[i] != ')':
            if '0' <= s[i] <= '9':
                cur = cur*10 + int(s[i])
                i += 1
            elif s[i] != '(':  # 遇到的是运算符号
                self.addNum(stack, cur)
                stack.append(s[i])
                i += 1
                cur = 0
            else:  # 遇到的是(
                arr = self.calc(s, i+1)
                cur = arr[0]
                print(cur)
                i = arr[1] + 1
        self.addNum(stack, cur)
        return [self.getNum(stack), i]

    def addNum(self, stack, num):
        """
        看一下，当前栈顶元素是什么运算符号，如果是 '+' or '-', 把num 直接放进去;
        如果是 '*' or '/', 则进行运算合并，将运算后的数字放到栈顶。
        :param stack:
        :param num:
        :return:
        """
        if stack:
            top = stack.pop()
            if top == '+' or top == '-':
                stack.append(top)
            else:
                cur = stack.pop()
                num = cur * num if top == '*' else cur / num
        stack.append(num)

    def getNum(self, stack):
        """
        对当前栈的元素进行运算
        :param stack:
        :return:
        """
        res = 0
        add = True
        while stack:
            cur = stack.pop(0)
            if cur == '+':
                add = True
            elif cur == '-':
                add = False
            else:
                res += cur if add else (-cur)

        return res




if __name__ == '__main__':
    # stack = MinStack2()
    # stack.push(-2)
    # stack.push(0)
    # stack.push(-3)
    # stack.push(5)
    # stack.push(-4)
    # print('最小元素：', stack.getMin())
    # stack.pop()
    # stack.pop()
    # stack.pop()
    # print('最小元素：', stack.getMin())
    # print(stack.minStack)'-2+35*((7+8)*(6-3))'

    obj = ExpressionCompute()
    exp = '-2+35*((7+8)*(6-3))'
    res = obj.solution(exp)
    print(eval(exp))
    print(res)

