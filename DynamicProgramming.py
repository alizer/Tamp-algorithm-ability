#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Name:         DynamicProgramming
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
                cur[j] += cur[j - 1]
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
    """
    https://leetcode-cn.com/problems/minimum-path-sum/ 最小路径和
    """

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


class CountBits:
    """
    https://leetcode-cn.com/problems/w3tCBm/
    剑指 Offer II 003. 前 n 个数字二进制中 1 的个数

    输入: n = 5
    输出: [0,1,1,2,1,2]
    解释:
    0 --> 0
    1 --> 1
    2 --> 10
    3 --> 11
    4 --> 100
    5 --> 101

    来源：力扣（LeetCode）
    链接：https://leetcode-cn.com/problems/w3tCBm
    著作权归领扣网络所有。商业转载请联系官方授权，非商业转载请注明出处。
    """

    def solution1(self, n: int) -> List[int]:
        """
        Brian Kernighan 算法
        :param n:
        :return:
        """

        def countones(x: int) -> int:
            ones = 0
            while x > 0:
                x &= (x - 1)
                ones += 1
            return ones

        bits = [countones(i) for i in range(n + 1)]

        return bits

    def solution2(self, n: int) -> List[int]:
        bits = [0]
        highBit = 0
        for i in range(1, n + 1):
            if i & (i - 1) == 0:
                highBit = i
            bits.append(bits[i - highBit] + 1)
        return bits


class SingleNumber:
    """
    https://leetcode-cn.com/problems/WGki4K/
    剑指 Offer II 004. 只出现一次的数字

    """

    def solution(self, nums: List[int]):
        ans = 0
        for i in range(32):
            total = sum((num >> i) & 1 for num in nums)
            if total % 3:
                # Python 这里对于最高位需要特殊判断
                if i == 31:
                    ans -= (1 << i)
                else:
                    ans |= (1 << i)
        return ans


class MaxProduct:
    def solution(self, words: List[str]) -> int:
        arr = [0] * len(words)
        for idx, word in enumerate(words):
            for c in word:
                # 1 左移的位数和0取逻辑或的结果，保存对应字母位为0时对应字母不存在，1则存在
                arr[idx] = arr[idx] | (1 << (ord(c) - ord('a')))

        res = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                # 对两两字符串的每一位取逻辑与的结果，若结果为1则说明有对应字母位同时为1，即两字符串存在相同字母
                if arr[i] & arr[j] == 0:
                    res = max(res, len(words[i]) * len(words[j]))

        return res


class CountSubstrings:
    """
    https://leetcode-cn.com/problems/a7VOhD/
    剑指 Offer II 020. 回文子字符串的个数
    """

    def solution(self, s: str) -> int:
        n = len(s)
        # dp[i][j] 表示第i个字符到第j个字符能否组成回文串
        dp = [[0] * n for _ in range(n)]

        ans = 0
        for i in range(n):
            dp[i][i] = 1
            ans += 1

        for i in range(n - 2, -1, -1):
            for j in range(i + 1, n, 1):
                if s[i] == s[j]:
                    if i < j - 1 and dp[i + 1][j - 1] == 1:
                        dp[i][j] = 1
                        ans += 1
                    elif i == j - 1:
                        dp[i][j] = 1
                        ans += 1
                else:
                    continue
        return ans


class RobotWalk:
    """
    机器人走路步数
    """

    def solution1(self, N, start, aim, K):
        """

        :param N: N个位置，记为1-N，N一定大于或者等于2
        :param start: 开始时机器人在其中的start位置上
        :param aim: 最终来到的位置
        :param K: 总过需要走K步
        :return: 多少种走法
        """
        if N < 2 or start < 1 or start > N or aim < 1 or aim > N or K < 1:
            return -1
        return self.process1(start, K, aim, N)

    def process1(self, cur, rest, aim, N):
        """

        :param cur: 机器人当前来到的位置是cur
        :param rest: 机器人还有rest步需要去走
        :param aim: 最终的目标是aim
        :param N:
        :return:
        """
        if rest == 0:
            return 1 if cur == aim else 0
        if cur == 1:
            return self.process1(2, rest - 1, aim, N)
        if cur == N:
            return self.process1(N - 1, rest - 1, aim, N)

        return self.process1(cur - 1, rest - 1, aim, N) + self.process1(cur + 1, rest - 1, aim, N)

    def solution2(self, N, start, aim, K):
        """
        递归 --> 记忆式搜索【弱动态规划】
        :param N:
        :param start:
        :param aim:
        :param K:
        :return:
        """
        if N < 2 or start < 1 or start > N or aim < 1 or aim > N or K < 1:
            return -1
        dp = [[-1] * (K + 1) for _ in range(N + 2)]
        return self.process2(start, K, aim, N, dp)

    def process2(self, cur, rest, aim, N, dp):
        """

        :param cur:
        :param rest:
        :param aim:
        :param N:
        :param dp: 空间换时间，减少重复计算
        :return:
        """
        if dp[cur][rest] != -1:
            return dp[cur][rest]
        # 之前没算过！
        if rest == 0:
            ans = 1 if cur == aim else 0
        elif cur == 1:
            ans = self.process2(2, rest - 1, aim, N, dp)
        elif cur == N:
            ans = self.process2(N - 1, rest - 1, aim, N, dp)
        else:
            ans = self.process2(cur - 1, rest - 1, aim, N, dp) + self.process2(cur + 1, rest - 1, aim, N, dp)

        # 更新数组
        dp[cur][rest] = ans
        return ans

    def solution3(self, N, start, aim, K):
        """
        记忆式搜索【弱动态规划】--> 强动态规划
        :param N:
        :param start:
        :param aim:
        :param K:
        :return:
        """
        if N < 2 or start < 1 or start > N or aim < 1 or aim > N or K < 1:
            return -1
        dp = [[0] * (K + 1) for _ in range(N + 1)]
        dp[aim][0] = 1
        for rest in range(1, K + 1):
            dp[1][rest] = dp[2][rest - 1]
            for cur in range(2, N):
                dp[cur][rest] = dp[cur - 1][rest - 1] + dp[cur + 1][rest - 1]
            dp[N][rest] = dp[N - 1][rest - 1]

        return dp[start][K]


class PredictWinners:
    """
    https://leetcode.cn/problems/predict-the-winner/
    """

    def solution1(self, arr: List[int]) -> int:
        """
        范围内 递归
        :param arr:
        :return: 返回获胜者的分数
        """
        if not arr:
            return 0
        # 先手
        first = self.f1(arr, 0, len(arr) - 1)
        # 后手
        second = self.g1(arr, 0, len(arr) - 1)
        return max(first, second)

    def f1(self, arr, L, R):
        if L == R:
            return arr[L]
        p1 = arr[L] + self.g1(arr, L + 1, R)
        p2 = arr[R] + self.g1(arr, L, R - 1)
        return max(p1, p2)

    def g1(self, arr, L, R):
        if L == R:
            return 0
        p1 = self.f1(arr, L + 1, R)
        p2 = self.f1(arr, L, R - 1)
        return min(p1, p2)

    def solution2(self, arr: List[int]):
        """
        暴力递归 --> 记忆化搜索
        :param arr:
        :return:
        """
        if not arr:
            return 0
        n = len(arr)
        fmap = [[-1] * n for _ in range(n)]
        gmap = [[-1] * n for _ in range(n)]

        first = self.f2(arr, 0, n - 1, fmap, gmap)
        second = self.g2(arr, 0, n - 1, fmap, gmap)
        return max(first, second)

    def f2(self, arr, L, R, fmap, gmap):
        """
        arr[L..R]，先手获得的最好分数返回
        :param arr:
        :return:
        """
        if fmap[L][R] != -1:
            return fmap[L][R]

        if L == R:
            ans = arr[L]
        else:
            p1 = arr[L] + self.g2(arr, L + 1, R, fmap, gmap)
            p2 = arr[R] + self.g2(arr, L, R - 1, fmap, gmap)
            ans = max(p1, p2)
        fmap[L][R] = ans
        return ans

    def g2(self, arr, L, R, fmap, gmap):
        """
        arr[L..R]，后手获得的最好分数返回
        :param arr:
        :param L:
        :param R:
        :param fmap:
        :param gmap:
        :return:
        """
        if gmap[L][R] != -1:
            return gmap[L][R]
        if L == R:
            ans = 0
        else:
            p1 = self.f2(arr, L + 1, R, fmap, gmap)
            p2 = self.f2(arr, L, R - 1, fmap, gmap)
            ans = min(p1, p2)
        gmap[L][R] = ans
        return ans

    def solution3(self, arr: List[int]):
        """
        记忆化搜索 --> 经典动态规划
        :param arr:
        :return:
        """
        if not arr:
            return 0
        n = len(arr)
        fmap = [[0] * n for _ in range(n)]
        gmap = [[0] * n for _ in range(n)]
        for i in range(n):
            fmap[i][i] = arr[i]

        for i in range(1, n):
            L, R = 0, i
            while R < n:
                fmap[L][R] = max(arr[L] + gmap[L + 1][R], arr[R] + gmap[L][R - 1])
                gmap[L][R] = min(fmap[L + 1][R], fmap[L][R - 1])
                L += 1
                R += 1

        return max(fmap[0][n - 1], gmap[0][n - 1])


class KnapSack:
    """
    背包问题
    """

    def solution1(self, w: List[int], v: List[int], bag):
        """
        从左到右递归，arr[0...i-1]已经决定好了，现在来到arr[i]位置，决定要还是不要
        :param w: 货物重量
        :param v: 货物价值
        :param bag: 背包容量，不能超过这个载重
        :return: 不超重的情况下，能够得到的最大价值
        """
        if not w or not v or bag <= 0 or len(w) != len(v):
            return 0
        return self.process1(w, v, 0, bag)

    def process1(self, w, v, index, rest):
        """
        从index开始到最后所能获得的最大货物价值，和0...index-1无关
        :param w:
        :param v:
        :param index:
        :param rest:
        :return:
        """
        # 剩余空间小于0了，无效的组合
        if rest < 0:
            return -1
        # 没有货物了，所以是0
        if index == len(w):
            return 0
        p1 = self.process1(w, v, index + 1, rest)
        p2 = 0
        next = self.process1(w, v, index + 1, rest - w[index])
        if next != -1:
            p2 = next + v[index]

        return max(p1, p2)

    def solution2(self, w: List[int], v: List[int], bag):
        """
        两个可变参数，index和rest，因此需要一个二维数组。
        从上到下，从左到右。
        动态规划， 二维数组dp，行位置表示物品索引【要和不要】；列位置表示
        剩余空间，范围为0~bag
        :param w:
        :param v:
        :param bag:
        :return:
        """
        if not w or not v or bag <= 0 or len(w) != len(v):
            return 0
        n = len(w)
        dp = [[0] * (bag + 1) for _ in range(n + 1)]
        for index in range(n - 1, -1, -1):
            for rest in range(bag + 1):
                p1 = dp[index + 1][rest]
                p2 = 0
                if rest - w[index] < 0:
                    next = -1
                else:
                    next = dp[index + 1][rest - w[index]]
                if next != -1:
                    p2 = next + v[index]

                dp[index][rest] = max(p1, p2)
        return dp[0][bag]


class NumDecodings:
    """
    https://leetcode.cn/problems/decode-ways/
    解码方法
    """

    def solution1(self, s: str) -> int:
        if not s:
            return 0
        return self.process1(s, 0)

    def process1(self, s: str, index: int) -> int:
        """
        s[0...index] 已转化完，不用关心
        s[index...]去转化，返回有多少种方法
        :param s:
        :param index:
        :return:
        """
        if index == len(s):
            return 1
        # i没到最后，说明有字符
        if str[index] == '0':  # 之前的决定有问题
            return 0
        # index 单独转
        ways = self.process1(s, index + 1)
        if index + 1 < len(s) and ((ord(s[index]) - ord('0')) * 10 + ord(s[index + 1]) - ord('0') < 27):
            ways += self.process1(s, index + 2)
        return ways

    def solution2(self, s: str) -> int:
        """
        记忆式搜索
        从右往左，动态规划
        dp[i]表示：str[i...]有多少种转化方式
        :param s:
        :return:
        """
        if not s:
            return 0

        n = len(s)
        dp = [0] * (n + 1)
        dp[n] = 1
        for index in range(n - 1, -1, -1):
            if s[index] != '0':
                ways = dp[index + 1]
                if index + 1 < len(s) and ((ord(s[index]) - ord('0')) * 10 + ord(s[index + 1]) - ord('0') < 27):
                    ways += dp[index + 2]

                dp[index] = ways

        return dp[0]

    def solution3(self, s: str) -> int:
        """
        从左往右，动态规划
        dp[i]表示：str[0...i]有多少种转化方式
        :param s:
        :return:
        """
        if not s or s[0] == '0':
            return 0

        n = len(s)
        dp = [0] * n
        dp[0] = 1

        for i in range(1, n):
            if s[i] == '0':
                """
                如果此时str[i]=='0'，那么他是一定要拉前一个字符(i-1的字符)一起拼的，
				那么就要求前一个字符，不能也是‘0’，否则拼不了。
				前一个字符不是‘0’就够了嘛？不够，还得要求拼完了要么是10，要么是20，如果更大的话，拼不了。
				这就够了嘛？还不够，你们拼完了，还得要求str[0...i-2]真的可以被分解！
				如果str[0...i-2]都不存在分解方案，那i和i-1拼成了也不行，因为之前的搞定不了。
                """
                if s[i - 1] == '0' or s[i - 1] > '2' or (i - 2 >= 0 and dp[i - 2] == 0):
                    return 0
                else:
                    dp[i] = dp[i - 2] if i - 2 >= 0 else 1

            else:
                dp[i] = dp[i - 1]
                if s[i - 1] != '0' and ((ord(s[i - 1]) - ord('0')) * 10 + ord(s[i]) - ord('0') < 27):
                    dp[i] += dp[i - 2] if i - 2 >= 0 else 1

        return dp[n - 1]


class minStickers:
    """
    https://leetcode.cn/problems/stickers-to-spell-word/
    贴纸拼词 所需最少贴纸
    """

    def solution1(self, stickers: List[str], target: str):
        """
        暴力递归
        :param stickers: 所有贴纸stickers，每一种贴纸都有无穷张
        :param target:
        :return: 最少张数
        """
        ans = self.process1(stickers, target)
        if ans == float('inf'):
            return -1
        else:
            return ans

    def process1(self, stickers, target):
        """
        当前剩余字符串为target，还需要的最少贴纸数量
        :param stickers:
        :param target:
        :return:
        """
        if len(target) == 0:
            return 0
        min_nums = float('inf')
        for sticker in stickers:
            rest = self.minus(target, sticker)
            if len(rest) != len(target):
                min_nums = min(min_nums, self.process1(stickers, rest))

        # 判断这一轮是否有用到贴纸，如果min_nums 状态没有改变，则没用到任何一张贴纸
        add_nums = 0 if min_nums == float('inf') else 1

        return min_nums + add_nums

    def minus(self, s1: str, s2: str) -> str:
        """
        用了s2这一张贴纸后，s1还剩下的字符串
        :param s1:
        :param s2:
        :return:
        """
        arr = [0] * 26
        for s in s1:
            arr[ord(s) - ord('a')] += 1

        for s in s2:
            arr[ord(s) - ord('a')] -= 1

        res = ''
        for i in range(len(arr)):
            if arr[i] > 0:
                for j in range(arr[i]):
                    res += chr(ord('a') + i)
        return res

    def solution2(self, stickers: List[str], target: str):
        """
        关键优化(用词频表替代贴纸数组)
        :param stickers:
        :param target:
        :return:
        """
        n = len(stickers)
        counts = [[0] * 26 for _ in range(n)]
        for i in range(n):
            for s in stickers[i]:
                counts[i][ord(s) - ord('a')] += 1
        ans = self.process2(counts, target)
        if ans == float('inf'):
            return -1
        else:
            return ans

    def process2(self, counts: List[List[int]], rest: str):
        """
        counts[i] 数组，当初i号贴纸的字符统计
        :param counts:
        :param rest:
        :return:
        """
        if len(rest) == 0:
            return 0
        tcounts = [0] * 26
        for s in rest:
            tcounts[ord(s) - ord('a')] += 1
        n = len(counts)
        min_nums = float('inf')
        for i in range(n):
            sticker = counts[i]
            # 最关键的优化(重要的剪枝!这一步也是贪心!)
            if sticker[ord(rest[0]) - ord('a')] > 0:
                sb = ''
                for j in range(26):
                    if tcounts[j] > 0:
                        nums = tcounts[j] - sticker[j]
                        if nums > 0:
                            sb += ''.join([chr(j + ord('a'))] * nums)
                min_nums = min(min_nums, self.process2(counts, sb))

        add_nums = 0 if min_nums == float('inf') else 1

        return min_nums + add_nums

    def solution3(self, stickers: List[str], target: str):
        """
        动态规划
        :param stickers:
        :param target:
        :return:
        """
        n = len(stickers)
        counts = [[0] * 26 for _ in range(n)]
        for i in range(n):
            for s in stickers[i]:
                counts[i][ord(s) - ord('a')] += 1
        dp = dict()
        dp[''] = 0
        ans = self.process3(counts, target, dp)
        if ans == float('inf'):
            return -1
        else:
            return ans

    def process3(self, counts, rest, dp):
        """
        引入词典，减少重复计算
        :param counts:
        :param rest:
        :param dp: key：剩余字符串；value：还需要的最小贴纸数
        :return:
        """
        if rest in dp.keys():
            return dp[rest]

        tcounts = [0] * 26
        for s in rest:
            tcounts[ord(s) - ord('a')] += 1
        n = len(counts)
        min_nums = float('inf')
        for i in range(n):
            sticker = counts[i]
            # 最关键的优化(重要的剪枝!这一步也是贪心!)
            if sticker[ord(rest[0]) - ord('a')] > 0:
                sb = ''
                for j in range(26):
                    if tcounts[j] > 0:
                        nums = tcounts[j] - sticker[j]
                        if nums > 0:
                            sb += ''.join([chr(j + ord('a'))] * nums)
                min_nums = min(min_nums, self.process3(counts, sb, dp))

        add_nums = 0 if min_nums == float('inf') else 1
        dp[rest] = min_nums + add_nums
        return min_nums + add_nums


class LongestCommonSubsequence:
    """
    最长公共子序列
    https://leetcode.cn/problems/longest-common-subsequence/
    """

    def solution1(self, s1, s2):
        """
        暴力递归
        :param s1:
        :param s2:
        :return:
        """
        if not s1 or not s2:
            return 0
        return self.process1(s1, s2, len(s1) - 1, len(s2) - 1)

    def process1(self, str1, str2, i, j):
        """
        str1[0...i]和str2[0...j]，这个范围上最长公共子序列长度是多少？
        可能性分类：
        a) 最长公共子序列，一定不以str1[i]字符结尾、也一定不以str2[j]字符结尾
        b) 最长公共子序列，可能以str1[i]字符结尾、但是一定不以str2[j]字符结尾
        c) 最长公共子序列，一定不以str1[i]字符结尾、但是可能以str2[j]字符结尾
        d) 最长公共子序列，必须以str1[i]字符结尾、也必须以str2[j]字符结尾
        注意：a)、b)、c)、d)并不是完全互斥的，他们可能会有重叠的情况
        但是可以肯定，答案不会超过这四种可能性的范围
        :param s1:
        :param s2:
        :param i:
        :param j:
        :return:
        """
        if i == 0 and j == 0:
            return 1 if str1[i] == str1[j] else 0
        elif i == 0:
            if str[i] == str[j]:
                return 1
            else:
                self.process1(str1, str2, i, j - 1)
        elif j == 0:
            if str[i] == str[j]:
                return 1
            else:
                self.process1(str1, str2, i - 1, j)
        else:  # 此时 i != 0 且 j != 0，str1[0...i]和str2[0...i]，str1和str2都不只一个字符
            # 最长公共子序列，一定不以str1[i]字符结尾、但是可能以str2[j]字符结尾
            p1 = self.process1(str1, str2, i - 1, j)
            # 最长公共子序列，可能以str1[i]字符结尾、但是一定不以str2[j]字符结尾
            p2 = self.process1(str1, str2, i, j - 1)
            if str1[i] == str2[j]:
                p3 = self.process1(str1, str2, i - 1, j - 1) + 1
                return max(p1, p2, p3)
            else:
                return max(p1, p2)

    def solution2(self, s1, s2):
        """
        动态规划
        :param s1:
        :param s2:
        :return:
        """
        if not s1 or not s2:
            return 0
        n, m = len(s1), len(s2)
        dp = [[0] * m for _ in range(n)]
        for i in range(n):
            dp[i][0] = 1 if s1[i] == s2[0] else dp[i - 1][0]
        for j in range(m):
            dp[0][j] = 1 if s1[0] == s2[j] else dp[0][j - 1]
        for i in range(1, n):
            for j in range(1, m):
                p1 = dp[i - 1][j]
                p2 = dp[i][j - 1]
                p3 = dp[i - 1][j - 1] + 1 if s1[i] == s2[j] else 0
                dp[i][j] = max(p1, p2, p3)
        return dp[n - 1][m - 1]


class LongestPalindromeSubseq:
    """
    最长回文子序列
    https://leetcode.cn/problems/longest-palindromic-subsequence/
    """

    def solution1(self, s: str) -> int:
        """
        暴力递归
        :param s:
        :return:
        """
        if not s:
            return 0
        return self.process1(s, 0, len(s) - 1)

    def process1(self, s, L, R):
        """
        s[L..R]最长回文子序列长度返回
        :param s:
        :param L:
        :param R:
        :return:
        """
        if L == R:
            return 1
        if L == R - 1:
            return 2 if s[L] == s[R] else 1
        # s[L]和s[R] 不相等，三种情况继续递归
        p1 = self.process1(s, L + 1, R - 1)
        p2 = self.process1(s, L, R - 1)
        p3 = self.process1(s, L + 1, R)

        p4 = self.process1(s, L + 1, R - 1) + 2 if s[L] == s[R] else 0

        return max(p1, p2, p3, p4)

    def solution2(self, s: str) -> int:
        """
        动态规划
        :param s:
        :return:
        """
        n = len(s)
        dp = [[0] * n for _ in range(n)]
        dp[n - 1][n - 1] = 1
        for i in range(n - 1):
            dp[i][i] = 1
            dp[i][i + 1] = 2 if s[i] == s[i + 1] else 1

        for i in range(n - 3, -1, -1):
            for j in range(i + 2, n):
                dp[i][j] = max(dp[i][j - 1], dp[i + 1][j])
                if s[i] == s[j]:
                    dp[i][j] = max(dp[i][j], 2 + dp[i + 1][j - 1])

        return dp[0][n - 1]


class SplitSumClosed:
    """
    给定一个正数数组arr，请把arr中所有的数分成两个集合，尽量让两个集合的累加和接近。
    返回：最接近的情况下，较小集合的累加和。
    """

    def solution(self, arr):
        """
        暴力递归
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return 0

        return self.process(arr, 0, sum(arr) // 2)

    def process(self, arr, i, rest):
        """
        arr[...i-1] 已经决定好了；
        arr[i...] 可以自由选择，返回累加和尽量接近rest，但不能超过rest的情况下，最接近的累加和是多少。
        :param arr:
        :param i:
        :param rest:
        :return:
        """
        if i == len(arr):
            return 0
        else:  # 还有数
            # 不使用arr[i]
            p1 = self.process(arr, i + 1, rest)
            p2 = 0
            # 使用arr[i]
            if arr[i] <= rest:
                p2 = arr[i] + self.process(arr, i + 1, rest - arr[i])

            return max(p1, p2)

    def solution1(self, arr):
        """
        动态规划
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return 0
        half_sum = sum(arr) // 2
        n = len(arr)
        dp = [[0] * (n + 1) for _ in range(half_sum + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(half_sum):
                p1 = dp[i + 1][j]
                p2 = 0
                if arr[i] <= j:
                    p2 = arr[i] + dp[i + 1][j - arr[i]]
                dp[i][j] = max(p1, p2)
        return dp[0][half_sum]


class SplitSumClosedSizedHalf:
    """
    给定一个正数数组arr，请把arr中所有的数分成两个集合，尽量让两个集合的累加和接近。
    要求：如果arr长度为偶数，两个集合包含数的个数要一样多；如果arr长度为奇数，两个集合包含数的个数必须只差一个。
    返回：最接近的情况下，较小集合的累加和。
    """

    def solution1(self, arr):
        if not arr or len(arr) < 2:
            return 0

        arr_sum = sum(arr) // 2
        # arr 长度为偶数
        if len(arr) & 1 == 0:
            return self.process(arr, 0, len(arr) / 2, arr_sum)
        else:
            one = self.process(arr, 0, len(arr) // 2, arr_sum)
            other = self.process(arr, 0, len(arr) // 2 + 1, arr_sum)
            return max(one, other)

    def process(self, arr, i, picks, rest):
        """
        arr[...i-1] 已经决定好了；
        arr[i....]自由选择，挑选的个数一定要是picks个，累加和<=rest, 离rest最近的返回。
        :param arr:
        :param i:
        :param picks:
        :param rest:
        :return:
        """
        if i == len(arr):
            # 有效的方案
            if picks == 0:
                return 0
            # 无效的方案
            else:
                return -1
        else:
            # 不要arr[i]
            p1 = self.process(arr, i + 1, picks, rest)
            # 要arr[i]
            p2 = -1
            next = -1
            if arr[i] <= rest:
                next = self.process(arr, i + 1, picks - 1, rest - arr[i])

            if next != -1:
                p2 = next + arr[i]
            return max(p1, p2)

    def solution2(self, arr):
        """
        动态规划
        :param arr:
        :return:
        """
        if not arr or len(arr) < 2:
            return 0

        half_sum = sum(arr) // 2
        n = len(arr)
        m = (n + 1) // 2
        dp = [[[-1] * (n + 1) for _ in range(m + 1)] for _ in range(half_sum + 1)]

        for rest in range(half_sum + 1):
            dp[n][0][rest] = 0

        for i in range(n - 1, -1, -1):
            for picks in range(m + 1):
                for rest in range(half_sum + 1):
                    # 不要arr[i]
                    p1 = dp[i + 1][picks][rest]
                    # 要arr[i]
                    p2 = -1
                    next = -1
                    if arr[i] <= rest and picks - 1 >= 0:
                        next = dp[i + 1][picks - 1][rest - arr[i]]
                    if next != -1:
                        p2 = next + arr[i]
                    dp[i][picks][rest] = max(p1, p2)

        # arr 长度为偶数
        if len(arr) & 1 == 0:
            return dp[0][n // 2][half_sum]
        else:
            return max(dp[0][n // 2 + 1][half_sum], dp[0][n // 2][half_sum])


class CordCoverMaxPoint:
    """
    给定一个有序数组arr，代表坐落在X轴上的点，给定一个正数K，代表绳子的长度，返回绳子最多压中几个点？
    即使绳子边缘处盖住点也算盖住
    """

    def solution(self, arr, K):
        """
        左右指针
        :param arr:
        :param K:
        :return:
        """
        l, r = 0, 0
        res = 0
        while r < len(arr):
            while l >= 0 and arr[r] - arr[l] <= K:
                l -= 1
            res = max(res, r - l)
            r += 1
        return res


class Drive:
    """
    现有司机N*2人，调度中心会将所有司机平分给A、B两区域，i号司机去A可得收入为income[i][0]，去B可得收入为income[i][1]
    返回能使所有司机总收入最高的方案是多少钱?
    """

    def solution(self, income: List[List[int]]):
        if not income or len(income) < 2 or len(income) & 1 != 0:
            return 0
        n = len(income)
        m = n >> 1
        self.process(income, 0, m)

    def process(self, income, index, rest):
        """
        暴力递归
        :param income:
        :param index: index.....所有的司机，往A和B区域分配！
        :param rest: A区域还有rest个名额!
        :return: 返回把index...司机，分配完，并且最终A和B区域同样多的情况下，index...这些司机，整体收入最大是多少！
        """
        if index == len(income):
            return 0
        # 还有司机待分配
        if len(income) - index == rest:
            # 全部分配给A司机
            return income[index][0] + self.process(income, index + 1, rest - 1)

        if rest == 0:
            # 全部分配给B司机
            return income[index][1] + self.process(income, index + 1, rest)

        # 当前司机，可以去A，或者去B
        p1 = income[index][0] + self.process(income, index + 1, rest - 1)
        p2 = income[index][1] + self.process(income, index + 1, rest)

        return max(p1, p2)

    def solution2(self, income):
        """
        动态规划
        :param income:
        :return:
        """
        if not income or len(income) < 2 or len(income) & 1 != 0:
            return 0

        n = len(income)
        m = n >> 1
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m + 1):
                if n - i == j:
                    # 全部分配给A司机
                    dp[i][j] = income[i][0] + dp[i + 1][j - 1]
                elif j == 0:
                    # 全部分配给B司机
                    dp[i][j] = income[i][1] + dp[i + 1][j]
                else:
                    # 当前司机，可以去A，或者去B
                    p1 = income[i][0] + dp[i + 1][j - 1]
                    p2 = income[i][1] + dp[i + 1][j]
                    dp[i][j] = max(p1, p2)

        return dp[0][m]

    def solution3(self, income):
        """
        贪心策略
        假设一共有10个司机，思路是先让所有司机去A，得到一个总收益
        然后看看哪5个司机改换门庭(去B)，可以获得最大的额外收益
        :param income:
        :return:
        """
        res = 0
        n = len(income)
        m = n >> 1
        arr = [0] * (n - 1)
        for i in range(n):
            res += income[i][0]
            arr[i] = income[i][0] - income[i][1]

        arr.sort()
        for i in range(n - 1, m - 1, -1):
            res += arr[i]

        return res


class FindTargetSumWays:
    """
    https://leetcode.cn/problems/target-sum/
    494. 目标和
    """

    def solution(self, arr, target):
        """
        暴力递归
        :param arr:
        :param target:
        :return:
        """
        return self.process(arr, 0, target)

    def process(self, arr, index, rest):
        if index == len(arr):
            return 1 if rest == 0 else 0

        return self.process(arr, index + 1, rest + arr[index]) + self.process(arr, index + 1, rest - arr[index])

    def solution2(self, arr, target):
        """
        记忆式搜索
        :param arr:
        :param target:
        :return:
        """
        dp = dict()
        return self.process2(arr, 0, target, dp)

    def process2(self, arr, index, rest, dp):
        # index == 7 rest = 13
        # map "7_13" 256
        if index == len(arr):
            return 1 if rest == 0 else 0

        if index in dp.keys() and rest in dp[index].keys():
            return dp[index][rest]

        if index == len(arr):
            return 1 if rest == 0 else 0
        else:
            ans = self.process(arr, index + 1, rest + arr[index]) + self.process(arr, index + 1, rest - arr[index])

        if index not in dp.keys():
            dp[index] = dict()
        else:
            dp[index][rest] = ans

        return ans

    def solution3(self, arr, target):
        """
        标准动态规划

        优化点一 :
        你可以认为arr中都是非负数
        因为即便是arr中有负数，比如[3,-4,2]
        因为你能在每个数前面用+或者-号
        所以[3,-4,2]其实和[3,4,2]达成一样的效果
        那么我们就全把arr变成非负数，不会影响结果的

        优化点二 :
        如果arr都是非负数，并且所有数的累加和是sum
        那么如果target<sum，很明显没有任何方法可以达到target，可以直接返回0

        优化点三 :
        arr内部的数组，不管怎么+和-，最终的结果都一定不会改变奇偶性
        所以，如果所有数的累加和是sum，
        并且与target的奇偶性不一样，没有任何方法可以达到target，可以直接返回0

        优化点四 :
        比如说给定一个数组, arr = [1, 2, 3, 4, 5] 并且 target = 3
        其中一个方案是 : +1 -2 +3 -4 +5 = 3
        该方案中取了正的集合为P = {1，3，5}
        该方案中取了负的集合为N = {2，4}
        所以任何一种方案，都一定有 sum(P) - sum(N) = target
        现在我们来处理一下这个等式，把左右两边都加上sum(P) + sum(N)，那么就会变成如下：
        sum(P) - sum(N) + sum(P) + sum(N) = target + sum(P) + sum(N)
        2 * sum(P) = target + 数组所有数的累加和
        sum(P) = (target + 数组所有数的累加和) / 2
        也就是说，任何一个集合，只要累加和是(target + 数组所有数的累加和) / 2
        那么就一定对应一种target的方式
        也就是说，比如非负数组arr，target = 7, 而所有数累加和是11
        求有多少方法组成7，其实就是求有多少种达到累加和(7+11)/2=9的方法

        优化点五 :
        二维动态规划的空间压缩技巧

        :param arr:
        :param target:
        :return:
        """
        sum_arr = 0
        for i in arr:
            sum_arr += abs(i)

        if sum_arr < target:
            return 0

        if (sum_arr & 1) ^ (target & 1) != 0:
            return 0

        return self.subset(arr, (target + sum_arr) >> 1)

    def subset(self, arr, s):
        """
        查找数组中累加和等于 s 的任意组合个数
        显然动态规划
        :param arr: 非负数组
        :param s:
        :return:
        """
        if s < 0:
            return 0
        dp = [0] * (s + 1)
        dp[0] = 1
        for n in arr:
            for i in range(s, n - 1, -1):
                dp[i] += dp[i - n]

        return dp[s]


class MinSwapStep:
    """
    GGBBGBGGG
    一个数组中只有两种字符'G'和'B'，可以让所有的G都放在左侧，所有的B都放在右侧
    或者可以让所有的G都放在右侧，所有的B都放在左侧，但是只能在相邻字符之间进行交换操作，返回至少需要交换几次
    """

    def solution(self, s):
        if not s:
            return 0

        step1, step2 = 0, 0
        gi, bi = 0, 0
        for i, c in enumerate(s):
            if c == 'G':  # 方案一，G去左边
                step1 += i - gi
                gi += 1
            else:  # 方案二，B去左边
                step2 += i - bi
                bi += 1
        return min(step1, step2)


class lengthOfLongestSubstring:
    """
    https://leetcode.cn/problems/longest-substring-without-repeating-characters/
    3. 无重复字符的最长子串 长度
    asdfabdf
    """

    def solution(self, s):
        if not s:
            return 0
        # ascii 码256个
        arr = [-1] * 256
        arr[ord(s[0])] = 0
        res = 1
        pre = 1

        for i in range(1, len(s)):
            pre = min(pre + 1, i - arr[ord(s[i])])
            res = max(res, pre)
            arr[ord(s[i])] = i

        return res


class IsInterleave:
    """
    输入：s1 = "aabcc", s2 = "dbbca", s3 = "aadbbcbcac"
    输出：true
    """

    def solution(self, s1, s2, s3):
        if not s1 and not s2 and not s3:
            return True
        if len(s1) + len(s2) != len(s3):
            return False

        m, n = len(s1), len(s2)
        dp = [[False] * (n + 1) for _ in range(m + 1)]

        dp[0][0] = True
        for i in range(1, m + 1):
            if s1[i - 1] != s3[i - 1]:
                break
            dp[i][0] = True

        for j in range(1, n + 1):
            if s2[j - 1] != s3[j - 1]:
                break
            dp[0][j] = True

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if (s3[i + j - 1] == s1[i - 1] and dp[i - 1][j]) or \
                        (s3[i + j - 1] == s2[j - 1] and dp[i][j - 1]):
                    dp[i][j] = True

        return dp[m][n]


class LengthOfLIS:
    """
    300. 最长递增子序列
    """

    def solution(self, nums: List[int]):
        """
        方案一: 动态规划
        辅助数组 dp，dp[i] 表示以nums[i]结尾的子序列的最大长度
        时间复杂度O(N^2)
        空间复杂度O(N)
        :param nums:
        :return:
        """
        n = len(nums)
        dp = [1] * n
        for i in range(1, n):
            tmp = 0
            for j in range(0, i):
                if nums[j] < nums[i]:
                    tmp = max(tmp, dp[j])
            dp[i] = tmp + 1
        return max(dp)

    def solution2(self, nums: List[int]):
        """
        方案二：动态规划
        辅助数组 ends，ends[i] 代表目前所有长度为i+1的递增子序列的最小结尾。
        可推断出 ends 也是递增序列
        时间复杂度O(N*logN)
        空间复杂度O(N)
        :param nums:
        :return:
        """
        if not nums:
            return 0

        n = len(nums)
        ends = [0] * n
        ends[0] = nums[0]
        l, r, m, right = 0, 0, 0, 0
        res = 1
        for i in range(1, n):
            l = 0
            r = right
            while l <= r:
                m = (l + r) // 2
                if nums[i] > ends[m]:
                    l = m + 1
                else:
                    r = m - 1
            right = max(right, l)
            ends[l] = nums[i]
            res = max(res, l + 1)

        return res

import megengine
class NCardsABWin:
    """
    谷歌面试题
	面值为1~10的牌组成一组，
	每次你从组里等概率的抽出1~10中的一张
	下次抽会换一个新的组，有无限组
	当累加和<17时，你将一直抽牌
	当累加和>=17且<21时，你将获胜
	当累加和>=21时，你将失败
	返回获胜的概率
    """

    def solution(self):
        return self.p1(0)

    def p1(self, cur: int):
        """
        当你来到cur这个累加和的时候，获胜概率是多少返回！
        :param cur:
        :return:
        """
        if 17 <= cur < 21:
            return 1.0
        if cur >= 21:
            return 0.0

        w = 0
        for i in range(10):
            w += self.p1(cur + i + 1)
        return w / 10

    def solution2(self, N, a, b):
        """
        谷歌面试题扩展版
	    面值为1~N的牌组成一组，
	    每次你从组里等概率的抽出1~N中的一张
	    下次抽会换一个新的组，有无限组
	    当累加和<a时，你将一直抽牌
	    当累加和>=a且<b时，你将获胜
	    当累加和>=b时，你将失败
	    返回获胜的概率，给定的参数为N，a，b
        :param N:
        :param a:
        :param b:
        :return:
        """
        if N < 1 or a >= b or a < 0 or b < 0:
            return 0.0
        if b - a >= N:
            return 1.0
        return self.p2(0, N, a, b)

    def p2(self, cur, N, a, b):
        """
        目前到达了cur的累加和, 返回赢的概率
        :param cur:
        :param N:
        :param a:
        :param b:
        :return:
        """
        if a <= cur < b:
            return 1.0
        if cur >= b:
            return 0.0

        w = 0
        for i in range(N):
            w += self.p2(cur + i + 1, N, a, b)
        return w / N

    def solution3(self, N, a, b):
        """
        针对solution3的优化，用到了观察位置优化枚举的技巧
        :param N:
        :param a:
        :param b:
        :return:
        """
        if N < 1 or a >= b or a < 0 or b < 0:
            return 0.0
        if b - a >= N:
            return 1.0
        return self.p2(0, N, a, b)

    def p3(self, cur, N, a, b):
        """
        目前到达了cur的累加和, 返回赢的概率
        :param cur:
        :param N:
        :param a:
        :param b:
        :return:
        """
        if a <= cur < b:
            return 1.0
        if cur >= b:
            return 0.0

        if cur == a-1:
            return 1.0 * (b-a)/N
        w = self.p3(cur+1, N, a, b) + self.p3(cur+1, N, a, b)*N

        if cur + 1 + N < b:
            w -= self.p3(cur+1+N, N, a, b)

        return w / N

    def solution4(self, N, a, b):
        """
        动态规划
        :param N:
        :param a:
        :param b:
        :return:
        """
        if N < 1 or a >= b or a < 0 or b < 0:
            return 0.0
        if b - a >= N:
            return 1.0
        dp = [0]*b
        for i in range(a, b):
            dp[i] = 1.0

        if a-1 >= 0:
            dp[a - 1] = 1.0 * (b-a)/N

        for cur in range(a-2, -1, -1):
            w = dp[cur + 1] + dp[cur + 1] * N

            if cur + 1 + N < b:
                w -= dp[cur + 1 + N]
            dp[cur] = w / N

        return dp[0]



if __name__ == '__main__':
    obj = NCardsABWin()
    # arr = [5, 7, 4, 5, 8, 1, 6, 0, 3, 4, 6, 1, 7]
    # weights = [3, 2, 4, 7, 8, 1, 7]
    # values = [5, 6, 3, 19, 12, 4, 2]
    res = obj.solution()
    res1 = obj.solution2(10, 16, 17)
    res2 = obj.solution3(10, 16, 17)
    res3 = obj.solution4(10, 16, 17)
    print(res1, res2, res3, sep='\n')
    # print(sum(arr))
    # bag = 15
    # res1 = obj.solution1(weights, values, bag)
    # res2 = obj.solution2(weights, values, bag)
    # res2 = obj.solution1(arr)
    # res3 = obj.solution1(arr)
    # res2 = obj.solution2(5, 2, 4, 6)
    # res3 = obj.solution3(5, 2, 4, 6)
    # print(res1, res2, res3)
    # obj = minStickers()
    # res = obj.minus('abcdf', 'bcfzzz')
    # print(res)

    # print(res1, res2)
