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


if __name__ == '__main__':
    obj = ThreeNumsSum()
    res = obj.threeSum([-1,0,1,2,-1,-4])
    print(res)