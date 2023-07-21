# import numpy as np


# 递归方式
def Levenshtein_Distance_Recursive(str1, str2):
    if len(str1) == 0:
        return len(str2)
    elif len(str2) == 0:
        return len(str1)
    elif str1 == str2:
        return 0
    if str1[len(str1) - 1] == str2[len(str2) - 1]:
        d = 0
    else:
        d = 1
    return min(
        Levenshtein_Distance_Recursive(str1, str2[:-1]) + 1,
        Levenshtein_Distance_Recursive(str1[:-1], str2) + 1,
        Levenshtein_Distance_Recursive(str1[:-1], str2[:-1]) + d,
    )


# 动态规划方式
def Levenshtein_Distance_DP(str1, str2):
    """
    矩阵是这样看的,
    比如 str1 = "ab", str2 = "adc"
       c a b
      0 1 2 3
    a 1 1 1 2
    b 2

    比如 第一个字符串的 a, 怎么变换成 c, ca, cab. 结果就是 1, 1, 2. 所以这就是第一行中的后三个值.
    """
    # 两个字符串的长度
    len1 = len(str1)
    len2 = len(str2)
    # dp = np.zeros((len1 + 1, len2 + 1))
    # 构建一个 (len1 + 1) * (len2 + 1) 的二维数组, 并初始化
    dp = [[0 for _ in range(len2 + 1)] for _ in range(len1 + 1)]
    # 遍历第一个字符串的每个位置, 包括空串
    for i in range(len1 + 1):
        # 初始化第一列, 表示从第一个字符串到空串的编辑距离
        dp[i][0] = i
    for j in range(len2 + 1):
        # 初始化第一行, 表示从空串到第二个字符串的编辑距离
        dp[0][j] = j

    # 从第一个字符串的第一个字符开始遍历
    for i in range(1, len1 + 1):
        # 从第二个字符串的第一个字符开始遍历
        for j in range(1, len2 + 1):
            # 如果两个字符相等, 则编辑距离为 0
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1
            # 计算当前位置的编辑距离, 取三种情况的最小值
            dp[i][j] = min(
                # 删除
                dp[i - 1][j] + 1,
                # 插入
                dp[i][j - 1] + 1,
                # 替换
                dp[i - 1][j - 1] + d,
            )
    # 返回最后一个位置的编辑距离
    return dp[len1][len2]


if __name__ == "__main__":
    text1 = "distance (str1, str2)，计算编辑距离"
    text2 = "hamming (str1, str2)，计算长度相等的字符串str1和str2的汉明距离"
    # print(Levenshtein_Distance_Recursive(text1, text2))
    print(Levenshtein_Distance_DP(text1, text2))
    print(Levenshtein_Distance_DP("ab", "cab"))
