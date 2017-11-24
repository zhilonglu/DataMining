#!/usr/bin/python
# coding=utf-8
from math import sqrt
import numpy as np

critics = {'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                         'Just My Luck': 3.0, 'Superman Returns': 3.5, 'You, Me and Dupree': 2.5,
                         'The Night Listener': 3.0},
           'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                            'Just My Luck': 1.5, 'Superman Returns': 5.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 3.5},
           'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                                'Superman Returns': 3.5, 'The Night Listener': 4.0},
           'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                            'The Night Listener': 4.5, 'Superman Returns': 4.0,
                            'You, Me and Dupree': 2.5},
           'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                            'Just My Luck': 2.0, 'Superman Returns': 3.0, 'The Night Listener': 3.0,
                            'You, Me and Dupree': 2.0},
           'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                             'The Night Listener': 3.0, 'Superman Returns': 5.0, 'You, Me and Dupree': 3.5},
           'Toby': {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 'Superman Returns': 4.0}}

"""
曼哈顿距离 计算相似度
"""


def manhattan_dix(x, y):
    return sum(abs(a - b) for a, b in zip(x, y))


"""
欧几里得空间法 计算相似度
"""


def sim_distance(p1, p2):
    c = set(p1.keys()) & set(p2.keys())
    if not c:
        return 0
    sum_of_squares = sum([pow(p1.get(sk) - p2.get(sk), 2) for sk in c])
    p = 1 / (1 + sqrt(sum_of_squares))
    return p


def eculidSim1(x, y):
    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


"""
明可夫斯基距离

当p==1,“明可夫斯基距离”变成“曼哈顿距离”
当p==2,“明可夫斯基距离”变成“欧几里得距离”
当p==∞,“明可夫斯基距离”变成“切比雪夫距离”
"""


def minkovski_dis(x, y, p):
    sumvalue = sum(pow(abs(a - b), p) for a, b in zip(x, y))
    mi = 1 / float(p)
    return round(sumvalue ** mi, 3)


"""
皮尔逊相关度
"""


def sim_distance_pearson(p1, p2):
    c = set(p1.keys()) & set(p2.keys())
    if not c:
        return 0
    s1 = sum([p1.get(sk) for sk in c])
    s2 = sum([p2.get(sk) for sk in c])
    sq1 = sum([pow(p1.get(sk), 2) for sk in c])
    sq2 = sum([pow(p2.get(sk), 2) for sk in c])
    ss = sum([p1.get(sk) * p2.get(sk) for sk in c])
    n = len(c)
    num = ss - s1 * s2 / n
    den = sqrt((sq1 - pow(s1, 2) / n) * (sq2 - pow(s2 , 2) / n))
    if den == 0:
        return 0
    p = num / den
    return p


def sim_pearson(prefs, p1, p2):
    # Get the list of mutually rated items
    si = {}
    for item in prefs[p1]:
        if item in prefs[p2]:
            si[item] = 1

            # if they are no ratings in common, return 0
    if len(si) == 0:
        return 0

        # Sum calculations
    n = len(si)

    # Sums of all the preferences
    sum1 = sum([prefs[p1][it] for it in si])
    sum2 = sum([prefs[p2][it] for it in si])

    # Sums of the squares
    sum1Sq = sum([pow(prefs[p1][it], 2) for it in si])
    sum2Sq = sum([pow(prefs[p2][it], 2) for it in si])

    # Sum of the products
    pSum = sum([prefs[p1][it] * prefs[p2][it] for it in si])

    # Calculate r (Pearson score)
    num = pSum - (sum1 * sum2 / n)
    den = sqrt((sum1Sq - pow(sum1, 2) / n) * (sum2Sq - pow(sum2, 2) / n))
    if den == 0:
        return 0

    r = num / den

    return r

    """
    Jaccard系数
    主要用于计算符号度量或布尔值度量的个体间的相似度
    """


def sim_distance_jaccard(p1, p2):
    c = set(p1.keys()) & set(p2.keys())
    if not c:
        return 0
    ss = sum([p1.get(sk) * p2.get(sk) for sk in c])
    sq1 = sum([pow(sk, 2) for sk in p1.values()])
    sq2 = sum([pow(sk, 2) for sk in p2.values()])
    p = float(ss) / (sq1 + sq2 - ss)
    return p


def jaccard_sim(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)


"""
余弦相似度
"""


def sim_distance_cos(p1, p2):
    c = set(p1.keys()) & set(p2.keys())
    if not c:
        return 0
    ss = sum([p1.get(sk) * p2.get(sk) for sk in c])
    sq1 = sqrt(sum([pow(p1.get(sk), 2) for sk in p1.keys()]))
    sq2 = sqrt(sum([pow(p2.get(sk), 2) for sk in p2.keys()]))
    p = float(ss) / (sq1 * sq2)
    return p


def cosine_dix(x, y):
    num = sum(map(float, x * y))
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return round(num / float(denom), 3)


"""
得到top相似度高的前几位
"""


def topMatches(prefs, person, n=5, similarity=sim_distance_pearson):
    scores = [similarity(prefs, person, other) for other in prefs if other != person]
    scores.sort()
    scores.reverse()
    return scores[0:n]


"""
#利用所有他人评价值加权平均,为某人提供建议.
"""


def getRecommendations(prefs, person, similarity=sim_distance):
    totals = {}
    simSums = {}

    for other in prefs:
        if other == person: continue
        sim = similarity(prefs, person, other)
        # 忽略评价值为0或小于0的情况.
        if sim <= 0: continue
        for item in prefs[other]:
            # 只对自己还未曾看过的影片进行评价.
            if item not in prefs[person] or prefs[person][item] == 0:
                totals.setdefault(item, 0)
                totals[item] += sim * prefs[other][item]
                # 相似度之和
                simSums.setdefault(item, 0)
                simSums[item] += sim
        # 建立一个归一化的列表.
        rankings = [(total / simSums[item], item) \
                    for item, total in totals.items()]
        rankings.sort()
        rankings.reverse()
        return rankings


if __name__ == '__main__':
    # print(sim_distance_cos(critics['Lisa'],critics['Tom']))
    # print(eculidSim1([1,3,2,4],[2,5,3,1]))#3.87298334621
    # print(manhattan_dix([1,3,2,4],[2,5,3,1]))#7
    # print(minkovski_dis([0,3,4,5],[7,6,3,-1],3))#8.373
    # print(cosine_dix(np.array([3,45,7,2]),np.array([2,54,13,15])))#0.972
    # print(jaccard_sim([0, 1, 2, 5, 6], [0, 2, 3, 5, 7, 9]))  # 0.375
    print(sim_pearson(critics, 'Lisa Rose', 'Gene Seymour'))
    print(sim_distance_pearson(critics['Lisa Rose'], critics['Gene Seymour']))
