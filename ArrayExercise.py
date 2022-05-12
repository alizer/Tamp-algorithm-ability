#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         ArrayExercise
# Author:       wendi
# Date:         2021/11/1

import heapq
from typing import List
import bisect

from DiversitySortAlgo import DiversitySort


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
        if not nums or len(nums) < 3:
            return ans

        for i in range(len(nums)):

            # 三数必有一负数
            if nums[i] > 0:
                break
            if i > 0 and nums[i] == nums[i - 1]:
                continue

            # 定义左右指针
            l = i + 1
            r = len(nums) - 1
            while l < r:
                sum = nums[i] + nums[l] + nums[r]
                if sum == 0:
                    ans.append([nums[i], nums[l], nums[r]])
                    while l < r and nums[l] == nums[l + 1]:
                        l += 1
                    while l < r and nums[r] == nums[r - 1]:
                        r -= 1
                    l += 1
                    r -= 1
                elif sum > 0:
                    r -= 1
                elif sum < 0:
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
        for i in range(length - 3):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            if nums[i] + nums[i + 1] + nums[i + 2] + nums[i + 3] > target:
                break
            if nums[i] + nums[length - 3] + \
                    nums[length - 2] + nums[length - 1] < target:
                continue

            for j in range(i + 1, length - 2):
                if j > i + 1 and nums[j] == nums[j - 1]:
                    continue
                if nums[i] + nums[j] + nums[j + 1] + nums[j + 2] > target:
                    break  # 这里直接跳出第二层循环
                if nums[i] + nums[j] + nums[length - 2] + \
                        nums[length - 1] < target:
                    continue

                l = j + 1
                r = length - 1
                while l < r:
                    tmp = nums[i] + nums[j] + nums[r] + nums[l]
                    if tmp == target:
                        res.append([nums[i], nums[j], nums[r], nums[l]])
                        while l < r and nums[r] == nums[r - 1]:
                            r -= 1
                        while l < r and nums[l] == nums[l + 1]:
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
            count += dc.get(pre_sum - k, 0)
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
            pre_sum.append(pre_sum[-1] + num)
        res = len(nums)
        for idx, num in enumerate(nums):
            if 2 * pre_sum[idx] + num == pre_sum[-1]:
                res = min(res, idx)

        return -1 if res == len(nums) else res


class NumMatrix:
    """
    剑指 Offer II 013. 二维子矩阵的和
    https://leetcode-cn.com/problems/O4NDxx/
    """

    def __init__(self, matrix: List[List[int]]):
        self.arr = [[0] * len(matrix[0]) for _ in range(len(matrix))]
        for i in range(len(self.arr)):
            for j in range(len(self.arr[0])):
                buf = 0
                if j >= 1:
                    buf = self.arr[i][j - 1]
                self.arr[i][j] = matrix[i][j] + buf

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:

        sum = 0
        for i in range(row1, row2 + 1, 1):
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
                return self.isPalindrome(
                    s,
                    left + 1,
                    right) or self.isPalindrome(
                    s,
                    left,
                    right - 1)
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
        arr1, arr2 = [0] * 26, [0] * 26

        if len(s) < len(p):
            return []

        for idx, c in enumerate(p):
            arr1[ord(c) - ord('a')] += 1
            arr2[ord(s[idx]) - ord('a')] += 1

        res = []
        for i in range(len(s) - len(p) + 1):
            if arr1 == arr2:
                res.append(i)
            if i + len(p) < len(s):
                arr2[ord(s[i + len(p)]) - ord('a')] += 1
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

            while rp <= len(s) - 1 and s[rp] not in tmp_set:
                tmp_set.add(s[rp])
                rp += 1
            max_len = max(max_len, rp - lp)
        return max_len


class MajorityElement(object):
    """
    169. 多数元素
    https://leetcode-cn.com/problems/majority-element/
    """

    def solution(self, arr: List[int]):
        """
        摩尔投票法
        时间复杂度：O(n)
        空间复杂度：O(1)
        :param arr:
        :return:
        """
        cand = None
        count = 0
        for item in arr:
            if count == 0:
                cand = item
                count += 1
                continue
            if cand == item:
                count += 1
            else:
                count -= 1
        return cand


class MajorityElement2(object):
    """
    229. 求众数 II
    https://leetcode-cn.com/problems/majority-element-ii/
    """

    def solution(self, arr: List[int]):
        """
        时间复杂度：O(n)
        空间复杂度：O(1)
        :param arr:
        :return:
        """
        cand1 = None
        cand2 = None
        count1 = 0
        count2 = 0
        res = []

        for item in arr:
            print(cand1, cand2)
            if cand1 == item:
                count1 += 1
                continue
            if cand2 == item:
                count2 += 1
                continue

            if count1 == 0:
                cand1 = item
                count1 += 1
                continue

            if count2 == 0:
                cand2 = item
                count2 += 1
                continue

            count1 -= 1
            count2 -= 1

        count1, count2 = 0, 0
        for item in arr:
            if cand1 == item:
                count1 += 1
            elif cand2 == item:
                count2 += 1

        if count1 > len(arr) / 3:
            res.append(cand1)

        if count2 > len(arr) / 3:
            res.append(cand2)

        return res


class ContainsNearbyAlmostDuplicate:
    """
    https://leetcode-cn.com/problems/7WqeDu/
    剑指 Offer II 057. 值和下标之差都在给定的范围内
    """

    def solution(self, nums: List[int], k: int, t: int) -> bool:
        """

        :param nums:
        :param k: 索引差值
        :param t: 元素差值
        :return:
        """
        def get_bucket_id(num, bucket_size):
            return ((num + 1) // bucket_size) - \
                1 if num < 0 else num // bucket_size

        bucket_map = dict()
        for i, num in enumerate(nums):
            bucket_id = get_bucket_id(num, t + 1)

            if bucket_id in bucket_map:
                return True
            elif bucket_id - 1 in bucket_map and abs(num - bucket_map[bucket_id - 1]) <= t:
                return True
            elif bucket_id + 1 in bucket_map and abs(num - bucket_map[bucket_id + 1]) <= t:
                return True

            bucket_map[bucket_id] = num

            if i - k >= 0:
                del_id = get_bucket_id(nums[i - k], t + 1)
                bucket_map.pop(del_id)

        return False


class BinarySearch:
    """
    二分查找,arr为有序列表
    """

    def solution(self, arr: List[int], num: int):
        """
        非递归思想，即在原有的划分的区域继续进行划分
        :param arr:
        :param num:
        :return:
        """
        l, r = 0, len(arr) - 1

        while l <= r:
            m = (l + r) // 2
            if arr[m] == num:
                return True
            elif arr[m] < num:
                l = m + 1
            else:
                r = m - 1
        return False

    def solution1(self, arr: List[int], num: int):
        """
        递归思想，在划分后的区域简历的list上，调用递归函数
        :param arr:
        :param num:
        :return:
        """
        n = len(arr)
        if n > 0:
            mid = n // 2
            if arr[mid] == num:
                return True
            elif arr[mid] < num:
                return self.solution1(arr[mid + 1:], num)
            else:
                return self.solution(arr[:mid], num)


class BinarySearchNearLeft:
    """
    返回有序数组 大于等于num的最左那个数的索引
    """
    def solution(self, arr, num):
        if not arr or len(arr) == 0:
            return -1

        l, r = 0, len(arr)-1
        res = -1
        while l <= r:
            mid = int((l+r) / 2)
            if arr[mid] >= num:
                res = mid
                r = mid - 1
            else:
                l = mid + 1
        return res


class BinarySearchLocalMin:
    """
    arr 相邻的数不相等，查找arr中任意一个局部最小值的位置：
    局部最小值定义：
    arr小于左边邻居且小于右边邻居，若只有左边邻居或右边邻居，
    只需要满足小于左边邻居或右边邻居即可；
    """
    def solution(self, arr):
        if not arr or len(arr) == 0:
            return -1
        n = len(arr)
        if n == 1:
            return 0
        if arr[0] < arr[1]:
            return 0
        if arr[n-1] < arr[n-2]:
            return n-1

        l, r = 0, n - 1
        # l, r 之间肯定有局部最小
        while l < r - 1:
            mid = int((l+r)/2)
            if arr[mid] < arr[mid-1] and arr[mid] < arr[mid+1]:
                return mid
            else:
                if arr[mid] > arr[mid - 1]:
                    r = mid - 1
                else:
                    l = mid + 1
        return l if arr[l] < arr[r] else r


class JumpToTargets:
    """
    跳到指定位置所需要的最小操作步数
    """
    def sum(self, m):
        return (m * (m+1))/2

    def solution(self, target):
        target = abs(target)
        l, r = 0, target
        m, near = 0, 0
        while l <= r:
            m = int((l + r)/2)
            if self.sum(m) >= target:
                near = m
                r = m - 1
            else:
                l = m + 1
        if self.sum(near) == target:
            return near

        # 差值为奇数  sum和规律 奇奇偶偶奇奇偶偶 交替，因此循环两次
        if ((self.sum(near) - target) & 1) == 1:
            near += 1

        if ((self.sum(near) - target) & 1) == 1:
            near += 1

        return near

class Avl:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.left, self.right = None, None


class MyCalendar:
    """
    https://leetcode-cn.com/problems/fi9suh/
    剑指 Offer II 058. 日程表
    """

    def __init__(self):
        self.allDate = None

        self.allDate1 = []

    def insert(self, root: Avl, start, end):
        if end <= root.start:
            if not root.left:
                root.left = Avl(start, end)
                return True
            return self.insert(root.left, start, end)
        elif start >= root.end:
            if not root.right:
                root.right = Avl(start, end)
                return True
            return self.insert(root.right, start, end)
        else:
            return False

    def book(self, start: int, end: int) -> bool:
        """
        平衡二叉树
        :param start:
        :param end:
        :return:
        """
        if not self.allDate:
            self.allDate = Avl(start, end)
            return True
        return self.insert(self.allDate, start, end)

    def book1(self, start: int, end: int) -> bool:
        """
        暴力搜索
        :param start:
        :param end:
        :return:
        """
        for arr in self.allDate1:
            if start < arr[1] and end > arr[0]:
                return False
        self.allDate1.append([start, end])
        return True


class KthLargest:

    def __init__(self, k: int, nums: List[int]):
        self.heap = []
        self.k = k
        for num in nums:
            heapq.heappush(self.heap, num)
            if len(self.heap) > k:
                heapq.heappop(self.heap)
        # print(self.heap)

    def add(self, val: int) -> int:
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]


class HeapObj():
    def __init__(self, k, v):
        self.k = k
        self.v = v

    def __lt__(self, other):
        if self.k < other.k:
            return True
        else:
            return False


class SplitNum:
    """
    给定一个数组arr和一个数num，将arr中小于num的数放在左边，等于num的数放在中间，
    大于arr的数放在右边。
    """
    def solution(self, arr, num):
        l, r = -1, len(arr)
        index = 0
        while index < r:
            if arr[index] < num:
                arr[l + 1], arr[index] = arr[index], arr[l + 1]
                l += 1
                index += 1
            elif arr[index] > num:
                arr[r - 1], arr[index] = arr[index], arr[r - 1]
                r -= 1
            else:
                index += 1
        return arr


if __name__ == '__main__':
    # KthLargest(k=3, nums=[])
    # obj = KthLargest(5, [4, 5, 8, 2, 3, 6, 7])
    obj = DiversitySort()
    # obj.insertSort(arr)
    arr = obj.generateRandomArr(20, 8)
    obj1 = SplitNum()
    print(obj1.solution(arr, 6))

    # print(obj.arr)
    # res = obj.solution(nums = [1,0,1,1], k = 1, t = 2)
    # print(res)
