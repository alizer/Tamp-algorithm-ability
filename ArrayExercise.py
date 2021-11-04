#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         ArrayExercise
# Author:       wendi
# Date:         2021/11/1

from typing import List


class ThreeNumsSum(object):
    """
    https://leetcode-cn.com/problems/3sum/
    三数之后
    """
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        """
        时间复杂度：O(n^2)，数组排序O(nlog(n))，双指针遍历 O(n^2)，总体O(nlog(n)) + O(n^2)，"留高次，取常数" O(n^2)
        空间复杂度：O(1)
        :param nums:
        :return:
        """
        ans = []
        nums.sort()
        if not nums or len(nums)<3:
            return ans

        for i in range(len(nums)):
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue
            l = i+1
            r = len(nums) - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum == 0:
                    ans.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l+1]:
                        l += 1
                    while l < r and nums[r] == nums[r-1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif sum>0:
                    r -= 1
                elif sum<0:
                    l += 1

        return ans

class ThreeNumsSumCloset(object):
    """
    https://leetcode-cn.com/problems/3sum-closest/
    最接近的三数之和
    """
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """
        时间复杂度：O(n^2)，数组排序O(nlog(n))，双指针遍历 O(n^2)，总体O(nlog(n)) + O(n^2)，"留高次，取常数" O(n^2)
        空间复杂度：O(1)
        :param nums:
        :param target:
        :return:
        """

        ans = nums[0] + nums[1] + nums[2]
        nums.sort()
        if not nums or len(nums) < 3:
            return ans

        for i in range(len(nums)):
            l = i + 1
            r = len(nums) - 1
            while l < r:
                tmp = nums[i] + nums[l] + nums[r]

                if abs(target - tmp) < abs(target - ans):
                    ans = tmp

                if tmp == target:
                    return tmp
                elif tmp > target:
                    r -= 1
                elif tmp < target:
                    l += 1

        return ans

class LetterCombfromNum(object):
    """
    https://leetcode-cn.com/problems/letter-combinations-of-a-phone-number/
    17. 电话号码的字母组合
    """
    def letterCombinations(self, digits: str) -> List[str]:
        """
        时间复杂度：
        空间复杂度：
        :param digits:
        :return:
        """
        if not digits:
            return []

        def comb_fun(arr1, arr2):
            res = []
            for i in arr1:
                for j in arr2:
                    res.append(i + j)
            return res

        dc = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        arr0 = ['']
        for digit in digits:
            arr1 = dc.get(digit)
            arr0 = comb_fun(arr0, arr1)
        return arr0

    def letterCombinations_1(self, digits: str) -> List[str]:
        """
        ***回溯法***
        时间复杂度：O(3^m+4^n),其中 m 是输入中对应 3 个字母的数字个数（包括数字 2、3、4、5、6、8），
        n 是输入中对应 4 个字母的数字个数（包括数字 7、9）
        空间复杂度：O(m+n)
        :param digits:
        :return:
        """


        if not digits:
            return list()

        phoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }

        def backtrack(index: int):
            if index == len(digits):
                combinations.append("".join(combination))
            else:
                digit = digits[index]
                for letter in phoneMap[digit]:
                    combination.append(letter)
                    backtrack(index + 1)
                    combination.pop()

        combination = list()
        combinations = list()
        backtrack(0)
        return combinations


class FourSum(object):
    """
    https://leetcode-cn.com/problems/4sum/
    """
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        """
        时间复杂度：O(n^3)
        空间复杂度：O(1)
        :param nums:
        :param target:
        :return:
        """
        if not nums or len(nums) < 4:
            return [[]]
        nums.sort()

        res, length = [], len(nums)
        for i in range(length-3):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[length - 3] + nums[length - 2] + nums[length - 1] < target:
                continue

            for j in range(i+1, length-2):
                if j > i+1 and nums[j] == nums[j-1]:
                    continue
                if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    break  # 这里直接跳出第二层循环
                if nums[i] + nums[j] + nums[length - 2] + nums[length - 1] < target:
                    continue

                l = j + 1
                r = length - 1
                while l < r:
                    tmp = nums[i] + nums[j] + nums[r] + nums[l]
                    if tmp == target:
                        res.append([nums[i], nums[j], nums[r], nums[l]])
                        while l < r and nums[r] == nums[r-1]:
                            r -= 1
                        while l < r and nums[l] == nums[l+1]:
                            l += 1
                        r -= 1
                        l += 1

                    elif tmp > target:
                        r -= 1
                    elif tmp < target:
                        l += 1
            else:
                continue
        return res

# class RemoveNthFromEnd(object):
#     """
#     https://leetcode-cn.com/problems/remove-nth-node-from-end-of-list/solution/hua-jie-suan-fa-19-shan-chu-lian-biao-de-dao-shu-d/
#     """
#     def solution(self, head: ):


if __name__ == '__main__':
    obj = FourSum()
    res = obj.fourSum(nums=[2,2,2,2,2], target = 8)
    print(res)