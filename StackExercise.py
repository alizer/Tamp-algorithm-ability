#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         StackExercise
# Author:       wendi
# Date:         2021/11/4


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


if __name__ == '__main__':
    stack = MinStack2()
    stack.push(-2)
    stack.push(0)
    stack.push(-3)
    stack.push(5)
    stack.push(-4)
    print('最小元素：', stack.getMin())
    stack.pop()
    stack.pop()
    stack.pop()
    print('最小元素：', stack.getMin())
    # print(stack.minStack)

