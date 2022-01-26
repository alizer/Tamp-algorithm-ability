#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         ArrayExercise
# Author:       wendi
# Date:         2021/11/1

from typing import List
import bisect


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


class MinSubArrayLen(object):
    """
    209. 长度最小的子数组
    https://leetcode-cn.com/problems/minimum-size-subarray-sum/
    """
    def solution(self, target: int, nums: List[int]) -> int:
        """
        时间复杂度：O(n)，其中 n 是数组的长度。指针 start 和 end 最多各移动 n 次。
        空间复杂度：O(1)
        :param target:
        :param nums:
        :return:
        """
        start, end = 0, 0
        n = len(nums)
        ans = n + 1
        curr_sum = 0

        while end < n:
            curr_sum += nums[end]
            while curr_sum >= target:
                ans = min(ans, end - start + 1)
                curr_sum -= nums[start]
                start += 1

            end += 1

        return 0 if ans > n else ans

    def solution2(self, s: int, nums: List[int]) -> int:
        if not nums:
            return 0

        n = len(nums)
        ans = n + 1
        sums = [0]
        for i in range(n):
            sums.append(sums[-1] + nums[i])

        for i in range(1, n + 1):
            target = s + sums[i - 1]
            bound = bisect.bisect_left(sums, target)
            if bound != len(sums):
                ans = min(ans, bound - (i - 1))

        return 0 if ans == n + 1 else ans

class NumSubarrayProductLessThanK():
    """
    https://leetcode-cn.com/problems/ZVAVXX/
    剑指 Offer II 009. 乘积小于 K 的子数组
    """
    def solution(self, nums: List[int], k: int) -> int:
        left, res = 0, 0
        prod = 1
        for right, num in enumerate(nums):
            prod *= num
            while left <= right and prod >= k:
                prod /= nums[left]
                left += 1
            if left <= right:
                res += right - left + 1

        return res


class SubarraySum(object):
    """
    https://leetcode-cn.com/problems/QTMn0o/
    剑指 Offer II 010. 和为 k 的子数组
    """
    def solution(self, nums: List[int], k: int) -> int:
        dc = {0: 1}
        pre_sum = 0
        count = 0
        for num in nums:
            pre_sum += num
            count += dc.get(pre_sum-k, 0)
            dc[pre_sum] = dc.get(pre_sum, 0) + 1

        return count


class findMaxLength(object):
    """
    https://leetcode-cn.com/problems/A1NYOS/
    剑指 Offer II 011. 0 和 1 个数相同的子数组
    """
    def solution(self, nums: List[int]) -> int:
        hashmap = {0: -1}
        pre_sum = 0
        max_len = 0
        for idx, num in enumerate(nums):
            if num == 1:
                pre_sum += 1
            else:
                pre_sum -= 1

            if pre_sum in hashmap:
                prefix_idx = hashmap.get(pre_sum)
                max_len = max(max_len, idx - prefix_idx)
            else:
                hashmap[pre_sum] = idx

        return max_len

class PivotIndex(object):
    """
    https://leetcode-cn.com/problems/tvdfij/
    剑指 Offer II 012. 左右两边子数组的和相等
    输入：nums = [1,7,3,6,5,6]
    输出：3
    """
    def solution(self, nums: List[int]) -> int:
        pre_sum = [0]
        for num in nums:
            pre_sum.append(pre_sum[-1]+num)
        res = len(nums)
        for idx, num in enumerate(nums):
            if 2*pre_sum[idx] + num == pre_sum[-1]:
                res = min(res, idx)

        return -1 if res == len(nums) else res

class NumMatrix:
    """
    剑指 Offer II 013. 二维子矩阵的和
    https://leetcode-cn.com/problems/O4NDxx/
    """

    def __init__(self, matrix: List[List[int]]):
        self.arr = [[0]*len(matrix[0]) for _ in range(len(matrix))]
        for i in range(len(self.arr)):
            for j in range(len(self.arr[0])):
                buf = 0
                if j >= 1:
                    buf = self.arr[i][j-1]
                self.arr[i][j] = matrix[i][j] + buf

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:

        sum = 0
        for i in range(row1, row2+1, 1):
            if col1 >= 1:
                sum += self.arr[i][col2] - self.arr[i][col1 - 1]
            else:
                sum += self.arr[i][col2]

        return sum

class CheckInclusion(object):
    """
    剑指 Offer II 014. 字符串中的变位词
    https://leetcode-cn.com/problems/MPnaiL/
    """
    def solution(self, s1: str, s2: str) -> bool:
        arr1, arr2, lg = [0] * 26, [0] * 26, len(s1)
        if lg > len(s2):
            return False

        for i in range(lg):
            arr1[ord(s1[i]) - ord('a')] += 1
            arr2[ord(s2[i]) - ord('a')] += 1

        for j in range(lg, len(s2)):
            if arr1 == arr2:
                return True
            arr2[ord(s2[j - lg]) - ord('a')] -= 1
            arr2[ord(s2[j]) - ord('a')] += 1

        return arr1 == arr2

class ValidPalindrome(object):
    """
    https://leetcode-cn.com/problems/RQku0D/
    剑指 Offer II 019. 最多删除一个字符得到回文
    """
    def __init__(self):
        pass

    def solution(self, s: str) -> bool:
        left = 0
        right = len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return self.isPalindrome(s, left+1, right) or self.isPalindrome(s, left, right-1)
            left += 1
            right -= 1
        return True

    def isPalindrome(self, s: str, left: int, right: int) -> bool:
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1

        return True


class FindAnagrams(object):
    """
    https://leetcode-cn.com/problems/VabMRr/
    剑指 Offer II 015. 字符串中的所有变位词
    """
    def solution(self, s: str, p: str) -> List[int]:
        arr1, arr2 = [0]*26, [0]*26

        if len(s)<len(p):
            return []

        for idx, c in enumerate(p):
            arr1[ord(c) - ord('a')] += 1
            arr2[ord(s[idx]) - ord('a')] += 1

        res = []
        for i in range(len(s)-len(p)+1):
            if arr1 == arr2:
                res.append(i)
            if i + len(p)<len(s):
                arr2[ord(s[i+len(p)]) - ord('a')] += 1
                arr2[ord(s[i]) - ord('a')] -= 1

        return res


class LengthOfLongestSubstring(object):
    """
    https://leetcode-cn.com/problems/wtcaE1/submissions/
    剑指 Offer II 016. 不含重复字符的最长子字符串
    """
    def solution(self, s: str) -> int:
        tmp_set = set()
        rp = 0
        max_len = 0

        for lp in range(len(s)):
            if lp > 0:
                tmp_set.remove(s[lp - 1])

            while rp <= len(s)-1 and s[rp] not in tmp_set:
                tmp_set.add(s[rp])
                rp += 1
            max_len = max(max_len, rp-lp)
        return max_len


if __name__ == '__main__':
    obj = NumMatrix(matrix=[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]])
    print(obj.arr)
    res = obj.sumRegion(2, 1, 4, 3)
    print(res)