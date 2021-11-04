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


if __name__ == '__main__':
    obj = ParenthesesValid()
    res = obj.isValid(']')
    print(res)
