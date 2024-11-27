import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv
import warnings
from pylab import mpl
from numpy import array, sign,zeros
from scipy.interpolate import interp1d
import scipy.signal
import pywt
import pywt.data
from mpl_toolkits.mplot3d import Axes3D
from operator import add
from functools import reduce
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing


# 处理warning
warnings.filterwarnings("ignore")  # 有时候代码处于某些原因会飘红却不影响正常的运行，为了美观使用该代码进行忽视处理

###########################读取CER数据
FILE_CER = "data.csv"
id_CER = []
data_CER = {}
with open(FILE_CER, "r", encoding="utf-8") as f:
    f_csv = csv.reader(f)
    for line in f_csv:
        if line[0] != "dataid":
            if line[0] in id_CER:
                data_CER[line[0]].append(line[1])
            else:
                data_CER[line[0]] = [line[1]]
                id_CER.append(line[0])
f.close()

for i in id_CER:
    for j in range(len(data_CER[i])):
        data_CER[i][j] = float(data_CER[i][j])

# 计数
lenSME = 0
lenRe = 0
other = 0
for id in id_CER:
    if id[0:2] == 'SM':
        lenSME = lenSME + 1

    if id[0:2] == 'Re':
        lenRe = lenRe + 1
print(FILE_CER)
print('SME用户数量：', lenSME)
print('Re用户数量：', lenRe)

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

# 设置plt绘图字体，中文宋体，英文Times New Roman
config = {
    "font.family": 'serif',
    "font.size": 14,
    "mathtext.fontset": 'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)

# 定义差分隐私
# 计算拉普拉斯分布的噪声
def laplace_noisy(sensitivety, epsilon):
    n_value = np.random.laplace(0, sensitivety/epsilon, 1)
    return n_value

# 计算拉普拉斯加噪后的值
def laplace_mech(data, sensitivity, epsilon):
    data1 = data
    for i in range(len(data1)):
        data1[i] += laplace_noisy(sensitivity, epsilon)
    return data1

# 定义sigmoid函数
def sigmoid(iX):
    for i in range(len(iX)):
        iX[i] = 1 / (1 + math.exp(-iX[i]))
    return iX

# 可调时延的电动汽车充电方波引入函数，返回电动汽车充电负荷，以及引入后总用户负荷
def EV_Charging_Intro(user_power, EV_power0, EV_power, time_lag):
    EV_power_lag = ([0] * time_lag) + EV_power
    k = []
    p = []
    user_power_EVIntro = [0] * len(user_power)
    EV_power_Intro = [0] * len(user_power)
    for b in range(0, len(EV_power_lag)):
        count1 = user_power[b] + EV_power_lag[b]
        count2 = EV_power0[b] + EV_power_lag[b]
        k.append(count1)
        p.append(count2)
    for r in range(0, len(user_power)):
        if r < len(k):
            user_power_EVIntro[r] = k[r]
            EV_power_Intro[r] = p[r]
        else:
            user_power_EVIntro[r] = user_power[r]
            EV_power_Intro[r] = EV_power0[r]
    return EV_power_Intro, user_power_EVIntro



'''
----------------------------------

             波动性性
      基于包络线面积的高频波动性
      基于SAX波动率的低频波动性

----------------------------------
'''


###############曲线小波分解出高低频分量并求二者占比
def plot_signal_decomp(data, w, title):
    """Decompose and plot a signal S.
    S = An + Dn + Dn-1 + ... + D1
    返回曲线高低频分量占比值alpha
    """
    w = pywt.Wavelet(w)#选取小波函数
    a = data
    ca = []#近似分量
    cd = []#细节分量
    for i in range(3):
        (a, d) = pywt.dwt(a, w, mode=pywt.Modes.smooth)#进行3阶离散小波变换
        ca.append(a)
        cd.append(d)

    rec_a = []
    rec_d = []

    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#重构

    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))

    # 合成分出低频分量和高频分量
    lowerfreq_curve = rec_a[2]
    higherfreq_curve = rec_d[0] + rec_d[1] + rec_d[2]

    # 计算高低频含量占比
    k = sum(abs(higherfreq_curve))/sum(abs(lowerfreq_curve))


    # ###作图
    # fig = plt.figure()
    # ax_main = fig.add_subplot(len(rec_a) + 1, 1, 1)
    # ax_main.set_title(title, font={'family':'Times New Roman', 'size':18})
    # ax_main.plot(data)
    # ax_main.set_xlim(0, len(data) - 1)
    #
    # for i, y in enumerate(rec_a):
    #     ax = fig.add_subplot(len(rec_a) + 1, 2, 3 + i * 2)
    #     ax.plot(y, 'r')
    #     ax.set_xlim((0, len(y) - 1))
    #     ax.set_ylabel("A%d" % (i + 1), font={'family':'Times New Roman', 'size':18})
    #
    # for i, y in enumerate(rec_d):
    #     ax = fig.add_subplot(len(rec_d) + 1, 2, 4 + i * 2)
    #     ax.plot(y, 'g')
    #     ax.set_xlim(0, len(y) - 1)
    #     ax.set_ylabel("D%d" % (i + 1), font={'family':'Times New Roman', 'size':18})



    return higherfreq_curve, lowerfreq_curve,k  # 返回高频曲线、低频曲线及高低频曲线含量占比


#################### 基于包络线面积的高频波动性

def general_equation(first_x, first_y, second_x, second_y):
    # 斜截式 y = kx + b
    A = second_y - first_y
    B = first_x - second_x
    C = second_x * first_y - first_x * second_y
    k = -1 * A / B
    b = -1 * C / B
    return k, b

def envelope_extraction(signal):  # 输入信号序列即可(list)
    s = signal.astype(float)
    q_u = np.zeros(s.shape)
    q_l = np.zeros(s.shape)

    # 在插值值前加上第一个值。这将强制模型对上包络和下包络模型使用相同的起点。
    # Prepend the first value of (s) to the interpolating values. This forces the model to use the same starting point for both the upper and lower envelope models.
    u_x = [0, ]  # 上包络的x序列
    u_y = [s[0], ]  # 上包络的y序列

    l_x = [0, ]  # 下包络的x序列
    l_y = [s[0], ]  # 下包络的y序列

    # 检测波峰和波谷，并分别标记它们在u_x,u_y,l_x,l_中的位置。
    # Detect peaks and troughs and mark their location in u_x,u_y,l_x,l_y respectively.

    for k in range(1, len(s) - 1):
        if (sign(s[k] - s[k - 1]) == 1) and (sign(s[k] - s[k + 1]) == 1):
            u_x.append(k)
            u_y.append(s[k])

        if (sign(s[k] - s[k - 1]) == -1) and ((sign(s[k] - s[k + 1])) == -1):
            l_x.append(k)
            l_y.append(s[k])

    u_x.append(len(s) - 1)  # 上包络与原始数据切点x
    u_y.append(s[-1])  # 对应的值

    l_x.append(len(s) - 1)  # 下包络与原始数据切点x
    l_y.append(s[-1])  # 对应的值

    # u_x,l_y是不连续的，以下代码把包络转为和输入数据相同大小的数组[便于后续处理，如滤波]
    upper_envelope_y = np.zeros(len(signal))
    lower_envelope_y = np.zeros(len(signal))

    upper_envelope_y[0] = u_y[0]  # 边界值处理
    upper_envelope_y[-1] = u_y[-1]
    lower_envelope_y[0] = l_y[0]  # 边界值处理
    lower_envelope_y[-1] = l_y[-1]

    # 上包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(u_x[0], u_y[0], u_x[1], u_y[1])  # 初始的k,b
    for e in range(1, len(upper_envelope_y) - 1):

        if e not in u_x:
            v = k * e + b
            upper_envelope_y[e] = v
        else:
            idx = u_x.index(e)
            upper_envelope_y[e] = u_y[idx]
            last_idx = u_x.index(e)
            next_idx = u_x.index(e) + 1
            # 求连续两个点之间的直线方程
            k, b = general_equation(u_x[last_idx], u_y[last_idx], u_x[next_idx], u_y[next_idx])

            # 下包络
    last_idx, next_idx = 0, 0
    k, b = general_equation(l_x[0], l_y[0], l_x[1], l_y[1])  # 初始的k,b
    for e in range(1, len(lower_envelope_y) - 1):

        if e not in l_x:
            v = k * e + b
            lower_envelope_y[e] = v
        else:
            idx = l_x.index(e)
            lower_envelope_y[e] = l_y[idx]
            last_idx = l_x.index(e)
            next_idx = l_x.index(e) + 1
            # 求连续两个切点之间的直线方程
            k, b = general_equation(l_x[last_idx], l_y[last_idx], l_x[next_idx], l_y[next_idx])

    return upper_envelope_y, lower_envelope_y


###################################基于SAX波动率的低频波动性
class SAX_trans:

    def __init__(self, ts, w, alpha):
        self.ts = ts
        self.w = w
        self.alpha = alpha
        self.aOffset = ord('a')  # 字符的起始位置，从a开始
        self.breakpoints = {'3': [-0.43, 0.43],
                            '4': [-0.67, 0, 0.67],
                            '5': [-0.84, -0.25, 0.25, 0.84],
                            '6': [-0.97, -0.43, 0, 0.43, 0.97],
                            '7': [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07],
                            '8': [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15],

                            }
        self.beta = self.breakpoints[str(self.alpha)]

    def normalize(self):  # 正则化
        X = np.asanyarray(self.ts)
        return (X - np.nanmean(X)) / np.nanstd(X)

    def paa_trans(self):  # 转换成paa
        tsn = self.normalize()  # 类内函数调用：法1：加self：self.normalize()   法2：加类名：SAX_trans.normalize(self)
        paa_ts = []
        n = len(tsn)
        xk = math.ceil(n / self.w)  # math.ceil()上取整，int()下取整
        for i in range(0, n, xk):
            temp_ts = tsn[i:i + xk]
            paa_ts.append(np.mean(temp_ts))
            i = i + xk
        return paa_ts

    def to_sax(self):  # 转换成sax的字符串表示
        tsn = self.paa_trans()
        len_tsn = len(tsn)
        len_beta = len(self.beta)
        strx = ''
        for i in range(len_tsn):
            letter_found = False
            for j in range(len_beta):
                if np.isnan(tsn[i]):
                    strx += '-'
                    letter_found = True
                    break
                if tsn[i] < self.beta[j]:
                    strx += chr(self.aOffset + j)
                    letter_found = True
                    break
            if not letter_found:
                strx += chr(self.aOffset + len_beta)
        return strx

    def compare_Dict(self):  # 生成距离表
        num_rep = range(self.alpha)  # 存放下标
        letters = [chr(x + self.aOffset) for x in num_rep]  # 根据alpha，确定字母的范围
        compareDict = {}
        len_letters = len(letters)
        for i in range(len_letters):
            for j in range(len_letters):
                if np.abs(num_rep[i] - num_rep[j]) <= 1:
                    compareDict[letters[i] + letters[j]] = 0
                else:
                    high_num = np.max([num_rep[i], num_rep[j]]) - 1
                    low_num = np.min([num_rep[i], num_rep[j]])
                    compareDict[letters[i] + letters[j]] = self.beta[high_num] - self.beta[low_num]
        return compareDict

    def dist(self, strx1, strx2):  # 求出两个字符串之间的mindist()距离值
        len_strx1 = len(strx1)
        len_strx2 = len(strx2)
        com_dict = self.compare_Dict()

        if len_strx1 != len_strx2:
            print("The length of the two strings does not match")
        else:
            list_letter_strx1 = [x for x in strx1]
            list_letter_strx2 = [x for x in strx2]
            mindist = 0.0
            for i in range(len_strx1):
                if list_letter_strx1[i] != '-' and list_letter_strx2[i] != '-':
                    mindist += (com_dict[list_letter_strx1[i] + list_letter_strx2[i]]) ** 2
            mindist = np.sqrt((len(self.ts) * 1.0) / (self.w * 1.0)) * np.sqrt(mindist)
            return mindist


def lowerfreq_volatility_jisuan(data):
    x1 = SAX_trans(ts=data, w=256, alpha=3)
    st1 = x1.to_sax()

    a = 0
    b = 0.3
    c = 0.7

    trance1 = {'a': [a, a, a], 'b': [b, b, b], 'c': [c, c, c]}  # 给各个符号赋值（待定）

    result = []
    for char in st1:
        for each in trance1[char]:
            result.append(each)

    sum_st1 = 0
    for i in range(len(st1) - 1):
        x = trance1[st1[i]][0]
        y = trance1[st1[i + 1]][0]
        sum_st1 = sum_st1 + abs(x - y)
    Preprivacy_lowfreq = sum_st1 / (len(st1) - 1)

    return result, Preprivacy_lowfreq



'''
----------------------------------

             自相关性
      基于R/S分析法的Hurst指数

----------------------------------
'''

def HalfSeries(s, n):
    '''
    if length(X) is odd:
        X <- {(X1 + X2) / 2, ..., (Xn-2 + Xn-1) / 2, Xn}
        n <- (n - 1) / 2
    else:
        X <- {(X1 + X2) / 2, ..., (Xn-1 + Xn) / 2}
        n <- n / 2
    return X, n
    '''
    X = []
    for i in range(0, len(s) - 1, 2):
        X.append((s[i] + s[i + 1]) / 2)
    # if length(s) is odd
    if len(s) % 2 != 0:
        X.append(s[-1])
        n = (n - 1) // 2
    else:
        n = n // 2
    return [np.array(X), n]


def Hurst(ts):
    '''
    Parameters
    ----------
    ts : Iterable Object.
        A time series or a list.

    Raises
    ------
    ValueError
        If input ts is not iterable then raise error.

    Returns
    -------
    H : Float
        The Hurst-index of this series.
    '''
    ts = np.array(ts)
    # N is use for storge the length sequence
    N, RS, n = [], [], len(ts)
    while (True):
        N.append(n)
        # Calculate the average value of the series
        m = np.mean(ts)
        # Construct mean adjustment sequence
        mean_adj = ts - m
        # Construct cumulative deviation sequence
        cumulative_dvi = np.cumsum(mean_adj)
        # Calculate sequence range
        srange = max(cumulative_dvi) - min(cumulative_dvi)
        # Calculate the unbiased standard deviation of this sequence
        unbiased_std_dvi = np.std(ts)
        # Calculate the rescaled range of this sequence under n length
        RS.append(srange / unbiased_std_dvi)
        # While n < 4 then break
        if n < 4:
            break
        # Rebuild this sequence by half-length
        ts, n = HalfSeries(ts, n)
    # Get Hurst-index by fit log(RS)~log(n)
    H = np.polyfit(np.log10(N), np.log10(RS), 1)[0]
    return H



'''
----------------------------------

            信息不确定性
              模糊熵

----------------------------------
'''

def FuzzyEn(s, r=0.2, m=2, n=2):
    '''s:需要计算熵的向量; r:阈值容限(标准差的系数); m:向量维数; n:模糊函数的指数
    '''
    N = len(s)  # 总长度
    th = r * np.std(s)  # 容限阈值

    def Phi(k):
        list_split = [s[i:i + k] for i in range(0, N - k + (k - m))]  # 将其拆分成多个子列表
        # 这里需要注意，2维和3维分解向量时的方式是不一样的！！！
        B = np.zeros(len(list_split))
        for i in range(0, len(list_split)):  # 遍历每个子向量
            di = np.abs(
                list_split[i] - np.mean(list_split[i]) - list_split + np.mean(list_split, 1).reshape(-1, 1)).max(1)
            Di = np.exp(- np.power(di, n) / th)
            B[i] = (np.sum(Di) - 1) / (len(list_split) - 1)  # 这里减1是因为要除去其本身，即exp(0)
        return np.sum(B) / len(list_split)

    H = - math.log(Phi(m + 1) / Phi(m))

    return H


'''
----------------------------------

           CER Dataset
            本征隐私性
             指标体系

----------------------------------
'''

# 定义计算本征隐私性InPI函数
def InPI(power_curve):

    # 归一化
    max_power = max(power_curve)
    min_power = min(power_curve)
    for n in range(len(power_curve)):
        power_curve[n] = (power_curve[n] - min_power) / (max_power - min_power)

    # 波动率
    higherfreq_curve, lowerfreq_curve, k = plot_signal_decomp(power_curve, 'sym5', 'Load Data')  # 分解高低频信号并求二者占比k
    upper_envelope_y, lower_envelope_y = envelope_extraction(np.array(power_curve))  # 上下包络线绘制
    higherfreq_volatility = np.trapz(upper_envelope_y - lower_envelope_y) / len(power_curve)  # 求取包络线面积，即高频波动性
    SAX_result, lowerfreq_volatility = lowerfreq_volatility_jisuan(power_curve)  # 计算低频波动性
    volatility = (((k / (1 + k)) * higherfreq_volatility + (1 / (1 + k)) * lowerfreq_volatility))  # 计算波动率

    # 非自相关率
    non_self_correlation = 1 - Hurst(power_curve)

    # 信息不确定性
    information_uncertainty = FuzzyEn(power_curve)

    # 本征隐私性向量
    InherentPrivacy = [0, 0, 0]  # 初始化本征隐私性向量
    InherentPrivacy[0] = volatility
    InherentPrivacy[1] = non_self_correlation
    InherentPrivacy[2] = information_uncertainty

    return InherentPrivacy, InherentPrivacy[0], InherentPrivacy[1], InherentPrivacy[2]


# 计算各用户InPr
fig = plt.figure()
ax = Axes3D(fig)
cSME = 0
cRe = 0
xSME = np.zeros(lenSME)
ySME = np.zeros(lenSME)
zSME = np.zeros(lenSME)
xRe = np.zeros(lenRe)
yRe = np.zeros(lenRe)
zRe = np.zeros(lenRe)


for id in id_CER:
    dataraw = data_CER[id][:768]

    # 计算本征隐私性InPI
    InherentPrivacy,InherentPrivacy[0], InherentPrivacy[1], InherentPrivacy[2] = InPI(dataraw)

    # 汇总最终结果
    if id[0:2] == 'SM':
        xSME[cSME] = InherentPrivacy[0]  # x坐标序列
        ySME[cSME] = InherentPrivacy[1]  # y坐标序列
        zSME[cSME] = InherentPrivacy[2]  # z坐标序列
        cSME = cSME + 1

    if id[0:2] == 'Re':
        xRe[cRe] = InherentPrivacy[0]  # x坐标序列
        yRe[cRe] = InherentPrivacy[1]  # y坐标序列
        zRe[cRe] = InherentPrivacy[2]  # z坐标序列
        cRe = cRe + 1

# 对最终结果进行归一化处理
xmax1 = max(xSME); xmax2 = max(xRe); xmax = max(xmax1, xmax2)
xmin1 = min(xSME); xmin2 = min(xRe); xmin = min(xmin1, xmin2)
ymax1 = max(ySME); ymax2 = max(yRe); ymax = max(ymax1, ymax2)
ymin1 = min(ySME); ymin2 = min(yRe); ymin = min(ymin1, ymin2)
zmax1 = max(zSME); zmax2 = max(zRe); zmax = max(zmax1, zmax2)
zmin1 = min(zSME); zmin2 = min(zRe); zmin = min(zmin1, zmin2)

for q in range(lenSME):
    xSME[q] = (xSME[q] - xmin) / (xmax - xmin)
    ySME[q] = (ySME[q] - ymin) / (ymax - ymin)
    zSME[q] = (zSME[q] - zmin) / (zmax - zmin)

for q in range(lenRe):
    xRe[q] = (xRe[q] - xmin) / (xmax - xmin)
    yRe[q] = (yRe[q] - ymin) / (ymax - ymin)
    zRe[q] = (zRe[q] - zmin) / (zmax - zmin)

# 输出归一化后的最终结果
x = np.append(xSME, xRe)
y = np.append(ySME, yRe)
z = np.append(zSME, zRe)
v = 0
InPr_SME_min = 1
InPr_Re_min = 1
InherentPrivacy = [0, 0, 0]

for id in id_CER:
    InherentPrivacy = [x[v], y[v], z[v]]
    # # 挑选本征隐私性最低的用户ID，后续引入EV
    # if id[0:2] == 'SM':
    #     InPr = (InherentPrivacy[0] + InherentPrivacy[1] + InherentPrivacy[2]) / 3
    #     if InPr < InPr_SME_min:
    #         InPr_SME_min = InPr
    #         print(id)
    # if id[0:2] == 'Re':
    #     InPr = (InherentPrivacy[0] + InherentPrivacy[1] + InherentPrivacy[2]) / 3
    #     if InPr < InPr_Re_min:
    #         InPr_Re_min = InPr
    #         print(id)
    # v = v + 1
    # print('用户', id, '本征隐私性=', InherentPrivacy)

    # # 输出归一化的标准id
    # if InherentPrivacy[0] == 1:
    #     print('xmax id:', id)
    #     print('原始值', xmax)
    # if InherentPrivacy[0] == 0:
    #     print('xmin id:', id)
    #     print('原始值', xmin)
    # if InherentPrivacy[1] == 1:
    #     print('ymax id:', id)
    #     print('原始值', ymax)
    # if InherentPrivacy[1] == 0:
    #     print('ymin id:', id)
    #     print('原始值', ymin)
    # if InherentPrivacy[2] == 1:
    #     print('zmax id:', id)
    #     print('原始值', zmax)
    # if InherentPrivacy[2] == 0:
    #     print('zmin id:', id)
    #     print('原始值', zmin)

    # # 输出归一化标准较高值id
    # if InherentPrivacy[0] > 0.9 or InherentPrivacy[1] > 0.9 or InherentPrivacy[2] > 0.9:
    #     print('id', id)

    v = v + 1

v = 0
InherentPrivacy = [0, 0, 0]
for id in id_CER:
    InherentPrivacy = [x[v], y[v], z[v]]
    # 输出劣质数据以供清洗
    if id[0:2] == 'SM':
        if InherentPrivacy[0] > 0.7 or InherentPrivacy[1] > 0.7 or InherentPrivacy[2] > 0.7:
            print(id)
    if id[0:2] == 'Re':
        if InherentPrivacy[0] < 0.3 or InherentPrivacy[1] < 0.3 or InherentPrivacy[2] < 0.3:
            print(id)

    v = v + 1

# 绘图
ax.scatter(xSME, ySME, zSME, color=(0, 0, 1))  # SME低本征隐私性 蓝色
ax.scatter(xRe, yRe, zRe, color=(1, 0, 0))  # Re低本征隐私性 红色

# ax.scatter(0.347, 0.407, 0.355, color=(0, 0, 1), s=100, marker='*')  # T-SME低本征隐私性
# ax.scatter(0.717, 0.754, 0.632, color=(1, 0, 0), s=100, marker='*')  # T-Re低本征隐私性

ax.scatter(0.5, 0.5, 0.5, c='black')  # 中心点
x_heidian=[0,0,0,0,0.5,0.5,0.5,1,1,1,1,0.5,0.5,0.5]
y_heidian=[0,0.5,0,0.5,0,0.5,0,1,0.5,0.5,1,1,0.5,1]
z_heidian=[0,0,0.5,0.5,0.5,0,0,1,0.5,1,0.5,0.5,1,1]
ax.scatter(x_heidian, y_heidian, z_heidian, c='black', s=3)  # 其余黑点

# 坐标轴设置
ax.grid()
ax.set_xlabel('Fluctuation', font={'family':'Times New Roman', 'size':14, 'weight':'bold'})
ax.set_xlim3d(0, 1)
ax.set_ylabel('Non-ACor', font={'family':'Times New Roman', 'size':14, 'weight':'bold'})
ax.set_ylim3d(0, 1)
ax.set_zlabel('Inf-Uncertainty', font={'family':'Times New Roman', 'size':14, 'weight':'bold'})
ax.set_zlim3d(0, 1)

plt.show()


'''
----------------------------------

            本征隐私性
             指标体系
            有效性验证
            
----------------------------------
'''

# # H-W算法预测如下，SARIMA算法预测、BP神经网络、LSTM算法见服务器
# Pred_Acc_HW = np.zeros(len(id_CER)) # H-W法预测准确率
# count = 0
# for id in id_CER:
#     dataraw = data_CER[id][672:768]
#     model = ExponentialSmoothing(data_CER[id][0:672], trend='add', seasonal='add', seasonal_periods=48).fit()
#     pred = model.forecast(96)
#
#     min_raw = np.min(dataraw)
#     max_raw = np.max(dataraw)
#     min_pred = np.min(pred)
#     max_pred = np.max(pred)
#
#     for n in range(96):
#         dataraw[n] = (dataraw[n] - min_raw)/(max_raw - min_raw)
#         pred[n] = (pred[n] - min_pred)/(max_pred - min_pred)
#
#     mse = mean_squared_error(pred, dataraw)
#     Pred_Acc_HW[count] = mse
#
#
#     # 输出劣质数据以供清洗
#     if id[0:2] == 'SM':
#         if Pred_Acc_HW[count] > 0.07:
#             print(id)
#     if id[0:2] == 'Re':
#         if Pred_Acc_HW[count] < 0.04:
#             print(id)
#
#     count = count + 1
#
# dataframe = pd.DataFrame({'HW': Pred_Acc_HW})
# dataframe.to_csv('HW.csv', index=False, sep=',')



'''
----------------------------------

        引入电动汽车充电
      观察对本征隐私性的影响

----------------------------------
'''

# # 电动汽车充电方波波形
# EV_slow_Fundamental = [3.5,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,3.5]  # 慢充功率7kW，慢充时间10h，即20个点
# EV_fast_Fundamental_Family = [20,20,20,20,20,20,20,20]# 家用快充20kW，充电时间4h，即8个点
# EV_fast_Fundamental = [100]  # 超级快充功率100kW，快充时间0.5h，即1个点(一般快充需求不是充满，而是救急)
#
# # 取单个用户用电功率曲线（一周时间，从周一0点开始，共336个点；其中【（0-16，38-48）+ i * 48】 19：00-次日8：00属于夜晚时间）  选取的用户ID为之前代码确定的最低InPr的用户ID
# user_power_Re = [a*2 for a in (data_CER['Re1273'][288:624])]
# user_power_SME = [a*10 for a in (data_CER['SME1145'][288:624])]
# # user_power_Re = data_CER['Re1008'][288:624]
# # user_power_SME = data_CER['SME1056'][288:624]
# blanklist = [0] * 336
#
#
# # 针对Re用户，在夜晚时间引入电动汽车慢充方波/快充方波
# Ori_Curve_Re = user_power_Re
# EV_power_slow = blanklist
# InPI_Re_Flu = [0] * 5
# InPI_Re_nonAC = [0] * 5
# InPI_Re_InfUn = [0] * 5
# InPI_Re_3to1 = [0] * 5  # 初始化  *场景数
#
# # 场景1：仅慢充每日充电时间相同
# user_power_Re = Ori_Curve_Re
# EV_power_slow = blanklist
# for i in range(6):  # 一周六晚引入电动汽车充电
#     EV_power_slow, user_power_Re = EV_Charging_Intro(user_power_Re, EV_power_slow, EV_slow_Fundamental, 38 + i * 48)  # 引入电动汽车后的一周内总功率曲线
#
# # plt.figure()
# # plt.plot(Ori_Curve_Re, color='grey', label='Original Load')  # 原负荷曲线
# # plt.plot(user_power_Re, color='tomato', label='Total Load after EV Charging')  # 引入电动汽车总负荷曲线
# # plt.plot(EV_power_slow, color='cornflowerblue', label='EV Charging')  # 电动汽车负荷曲线
# # plt.title('Re Scenario 1')
# # plt.legend()
# # plt.xlabel('Time')
# # plt.ylabel('Load')
# #
# # # 保存为Excel文件
# # data = {'Ori_Curve_Re': Ori_Curve_Re, 'user_power_Re': user_power_Re, 'EV_charge_power': EV_power_slow}
# # df = pd.DataFrame(data)
# # df.to_excel('1.xlsx', index=False)
#
# InPI_Re_Scenario1, InPI_Re_Flu[1], InPI_Re_nonAC[1], InPI_Re_InfUn[1] = InPI(user_power_Re)
#
# # 场景2：每日充电时间相同快充
# user_power_Re = Ori_Curve_Re
# EV_power_slow = blanklist
# for i in range(6):  # 一周六晚引入电动汽车充电，每天时延
#     EV_power_slow, user_power_Re = EV_Charging_Intro(user_power_Re, EV_power_slow, EV_fast_Fundamental_Family, 38 + i * 48)  # 引入电动汽车后的一周内总功率曲线
#
# # plt.figure()
# # plt.plot(Ori_Curve_Re, color='grey', label='Original Load')  # 原负荷曲线
# # plt.plot(user_power_Re, color='tomato', label='Total Load after EV Charging')  # 引入电动汽车总负荷曲线
# # plt.plot(EV_power_slow, color='cornflowerblue', label='EV Charging')  # 电动汽车负荷曲线
# # plt.title('Re Scenario 2')
# # plt.legend()
# # plt.xlabel('Time')
# # plt.ylabel('Load')
# #
# #
# # # 保存为Excel文件
# # data = {'Ori_Curve_Re': Ori_Curve_Re, 'user_power_Re': user_power_Re, 'EV_charge_power': EV_power_slow}
# # df = pd.DataFrame(data)
# # df.to_excel('2.xlsx', index=False)
#
#
# InPI_Re_Scenario1, InPI_Re_Flu[2], InPI_Re_nonAC[2], InPI_Re_InfUn[2] = InPI(user_power_Re)
#
# # 场景3：每晚随机时间家用快充
# Sum_Flu = 0
# Sum_nonAC = 0
# Sum_InfUn = 0
# for k in range(100):  # 随机情况重复1000次取均值
#     user_power_Re = Ori_Curve_Re
#     EV_power_fast = blanklist
#     for i in range(6):
#         EV_power_fast, user_power_Re = EV_Charging_Intro(user_power_Re, EV_power_fast, EV_fast_Fundamental_Family,
#                                                          38 + i + i * 48)  # 引入电动汽车后的一周内总功率曲线
#
#     # if k == 0:  # 只画一次随机情况的负荷曲线图
#     #     plt.figure()
#     #     plt.plot(Ori_Curve_Re, color='grey', label='Original Load')  # 原负荷曲线
#     #     plt.plot(user_power_Re, color='tomato', label='Total Load after EV Charging')  # 引入电动汽车总负荷曲线
#     #     plt.plot(EV_power_fast, color='cornflowerblue', label='EV Charging')  # 电动汽车负荷曲线
#     #     plt.title('Re Scenario 3')
#     #     plt.legend()
#     #     plt.xlabel('Time')
#     #     plt.ylabel('Load')
#     #
#     #     # 保存为Excel文件
#     #     data = {'Ori_Curve_Re': Ori_Curve_Re, 'user_power_Re': user_power_Re, 'EV_charge_power': EV_power_slow}
#     #     df = pd.DataFrame(data)
#     #     df.to_excel('3.xlsx', index=False)
#
#     InPI_Re_Scenario1, InPI_Re_Flu[3], InPI_Re_nonAC[3], InPI_Re_InfUn[3] = InPI(user_power_Re)
#     Sum_Flu = Sum_Flu + InPI_Re_Flu[3]
#     Sum_nonAC = Sum_nonAC + InPI_Re_nonAC[3]
#     Sum_InfUn = Sum_InfUn + InPI_Re_InfUn[3]
# InPI_Re_Flu[3] = Sum_Flu / 100
# InPI_Re_nonAC[3] = Sum_nonAC / 100
# InPI_Re_InfUn[3] = Sum_InfUn / 100
#
#
# # 场景4：每晚随机时间随机快慢充
# Sum_Flu = 0
# Sum_nonAC = 0
# Sum_InfUn = 0
# for k in range(100):
#     user_power_Re = Ori_Curve_Re
#     EV_power_fast = blanklist
#
#     for i in range(6):
#         c = random.randint(0, 1)  # 随机0/1， 0慢充，1快充
#         if c == 0:
#             EV_power_fast, user_power_Re = EV_Charging_Intro(user_power_Re, EV_power_fast, EV_slow_Fundamental,
#                                                              random.randint(38, 44) + i * 48)  # 引入电动汽车后的一周内总功率曲线
#         if c == 1:
#             EV_power_fast, user_power_Re = EV_Charging_Intro(user_power_Re, EV_power_fast, EV_fast_Fundamental_Family,
#                                                              random.randint(38, 54) + i * 48)  # 引入电动汽车后的一周内总功率曲线
#
#     # if k == 0:
#     #     plt.figure()
#     #     plt.plot(Ori_Curve_Re, color='grey', label='Original Load')  # 原负荷曲线
#     #     plt.plot(user_power_Re, color='tomato', label='Total Load after EV Charging')  # 引入电动汽车总负荷曲线
#     #     plt.plot(EV_power_fast, color='cornflowerblue', label='EV Charging')  # 电动汽车负荷曲线
#     #     plt.title('Re Scenario 4')
#     #     plt.legend()
#     #     plt.xlabel('Time')
#     #     plt.ylabel('Load')
#     #
#     #     # 保存为Excel文件
#     #     data = {'Ori_Curve_Re': Ori_Curve_Re, 'user_power_Re': user_power_Re, 'EV_charge_power': EV_power_fast}
#     #     df = pd.DataFrame(data)
#     #     df.to_excel('4.xlsx', index=False)
#
#     InPI_Re_Scenario1, InPI_Re_Flu[4], InPI_Re_nonAC[4], InPI_Re_InfUn[4] = InPI(user_power_Re)
#     Sum_Flu = Sum_Flu + InPI_Re_Flu[4]
#     Sum_nonAC = Sum_nonAC + InPI_Re_nonAC[4]
#     Sum_InfUn = Sum_InfUn + InPI_Re_InfUn[4]
# InPI_Re_Flu[4] = Sum_Flu / 100
# InPI_Re_nonAC[4] = Sum_nonAC / 100
# InPI_Re_InfUn[4] = Sum_InfUn / 100
#
#
# # 计算未引入电动汽车原始曲线本征隐私性
# InPI_Re_Scenario1, InPI_Re_Flu[0], InPI_Re_nonAC[0], InPI_Re_InfUn[0] = InPI(Ori_Curve_Re)  # 特别注意运行InPI代码后变量曲线会被归一化
#
#
# # 本征隐私性指标归一化
# # 求各指标值的最大值以便后续归一化（取极端条件即白噪声作为最值）
#
# # sum_Flu = 0
# # sum_nonAC = 0
# # sum_InfUn = 0
# # for i in range(1000):
# #     list = [(random.randint(0,1000) / 100) for k in range(336)]
# #     a, b, c, d = InPI(list)
# #     sum_Flu = sum_Flu + b
# #     sum_nonAC = sum_nonAC + c
# #     sum_InfUn = sum_InfUn + d
# #
# # max_InPI_Flu = sum_Flu / 1000
# # max_InPI_nonAC = 0.5
# # max_InPI_InfUn = sum_InfUn / 1000
#
# # ###由以上代码确定最终各指标值归一化最大最小值计算结果 max_InPI_Flu = 0.36; max_InPI_nonAC = 0.5; max_InPI_InfUn = 0.876
#
#
# max_InPI_Flu = 0.36
# min_InPI_Flu = 0
# for n in range(len(InPI_Re_Flu)):
#     InPI_Re_Flu[n] = (InPI_Re_Flu[n] - min_InPI_Flu) / (max_InPI_Flu - min_InPI_Flu)
#
# max_InPI_nonAC = 0.5
# min_InPI_nonAC = 0
# for n in range(len(InPI_Re_nonAC)):
#     InPI_Re_nonAC[n] = (InPI_Re_nonAC[n] - min_InPI_nonAC) / (max_InPI_nonAC - min_InPI_nonAC)
#
# max_InPI_InfUn = 0.876
# min_InPI_InfUn = 0
# for n in range(len(InPI_Re_InfUn)):
#     InPI_Re_InfUn[n] = (InPI_Re_InfUn[n] - min_InPI_InfUn) / (max_InPI_InfUn - min_InPI_InfUn)
#
#
#
# # 绘图观察本征隐私性变化
# x = [0, 1, 2, 3, 4]
# xvalues = ['0', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'Ⅳ']
# # plt.figure()
# # plt.plot(InPI_Re_Flu)
# # plt.title('Fluctuation')
# # plt.xlabel('场景')
# # plt.xticks(x, xvalues)
# # plt.figure()
# # plt.plot(InPI_Re_nonAC)
# # plt.title('nonACor')
# # plt.xlabel('场景')
# # plt.xticks(x, xvalues)
# # plt.figure()
# # plt.plot(InPI_Re_InfUn)
# # plt.title('InfUncertainty')
# # plt.xlabel('场景')
# # plt.xticks(x, xvalues)
#
# # 计算InPI三合一指标变化
# for k in range(len(InPI_Re_3to1)):
#     InPI_Re_3to1[k] = 0.25*InPI_Re_Flu[k] + 0.5*InPI_Re_nonAC[k] + 0.25*InPI_Re_InfUn[k]
# print(InPI_Re_3to1)
# # plt.figure()
# # plt.plot(InPI_Re_3to1)
# # plt.xlabel('Re Scenario')
# # plt.ylabel('InPr')
# # plt.xticks(x, xvalues)
# #
# # plt.show()


#
# # 针对SME用户，在白天时间引入电动汽车慢充方波/快充方波；其中【（18-36）+ i * 48】 8：00-19：00属于白天上班时间
#
# # 假设共有五台电动汽车，可在工作日白天任意时间进行慢充、快充和超级快充
#
# Ori_Curve_SME = user_power_SME
# EV_charge_power = blanklist
# InPI_SME_Flu = [0] * 5
# InPI_SME_nonAC = [0] * 5
# InPI_SME_InfUn = [0] * 5
# InPI_SME_3to1 = [0] * 5  # 初始化  *场景数
#
# # 场景1：随机数量汽车慢充，充电时间随机
# user_power_SME = Ori_Curve_SME
# EV_charge_power = blanklist
# for k in range(5):  # 五个工作日白天
#     EV_c = random.randint(0, 5)  # 每天随机充电汽车数量
#     for i in range(EV_c):  # 随机汽车数量
#         EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power, EV_slow_Fundamental, random.randint(16, 18) + k * 48)  # 引入电动汽车后的一周内总功率曲线
#
# plt.figure()
# plt.plot(Ori_Curve_SME, color='grey', label='原负荷曲线')  # 原负荷曲线
# plt.plot(user_power_SME, color='tomato', label='引入电动汽车总负荷曲线')  # 引入电动汽车总负荷曲线
# plt.plot(EV_charge_power, color='cornflowerblue', label='电动汽车充电负荷曲线')  # 电动汽车负荷曲线
# plt.title('SME场景1：随机汽车数量随机时间慢充')
# plt.legend()
# plt.xlabel('时间')
# plt.ylabel('负荷')
#
# # 创建DataFrame
# data = {'Ori_Curve_SME': Ori_Curve_SME, 'user_power_SME': user_power_SME, 'EV_charge_power': EV_charge_power}
# df = pd.DataFrame(data)
#
# # 保存为Excel文件
# df.to_excel('1.xlsx', index=False)
#
#
#
# InPI_SME_Scenario, InPI_SME_Flu[1], InPI_SME_nonAC[1], InPI_SME_InfUn[1] = InPI(user_power_SME)
#
# # 场景2：全部汽车家用快充，充电时间随机
# Sum_Flu = 0
# Sum_nonAC = 0
# Sum_InfUn = 0
# for l in range(1000):
#     user_power_SME = Ori_Curve_SME
#     EV_charge_power = blanklist
#
#     for k in range(5):  # 五台电动汽车
#         for i in range(5):  # 五个工作日白天
#             EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                 EV_fast_Fundamental_Family,
#                                                                 random.randint(16, 30) + i * 48)  # 引入电动汽车后的一周内总功率曲线
#
#     if l == 1:
#         plt.figure()
#         plt.plot(Ori_Curve_SME, color='grey', label='原负荷曲线')  # 原负荷曲线
#         plt.plot(user_power_SME, color='tomato', label='引入电动汽车总负荷曲线')  # 引入电动汽车总负荷曲线
#         plt.plot(EV_charge_power, color='cornflowerblue', label='电动汽车充电负荷曲线')  # 电动汽车负荷曲线
#         plt.title('SME场景2：全部汽车随机时间家用快充')
#         plt.legend()
#         plt.xlabel('时间')
#         plt.ylabel('负荷')
#
#         # 创建DataFrame
#         data = {'Ori_Curve_SME': Ori_Curve_SME, 'user_power_SME': user_power_SME, 'EV_charge_power': EV_charge_power}
#         df = pd.DataFrame(data)
#
#         # 保存为Excel文件
#         df.to_excel('2.xlsx', index=False)
#
#     InPI_SME_Scenario, InPI_SME_Flu[2], InPI_SME_nonAC[2], InPI_SME_InfUn[2] = InPI(user_power_SME)
#
#     Sum_Flu = Sum_Flu + InPI_SME_Flu[2]
#     Sum_nonAC = Sum_nonAC + InPI_SME_nonAC[2]
#     Sum_InfUn = Sum_InfUn + InPI_SME_InfUn[2]
# InPI_SME_Flu[2] = Sum_Flu / 1000
# InPI_SME_nonAC[2] = Sum_nonAC / 1000
# InPI_SME_InfUn[2] = Sum_InfUn / 1000
#
#
# # 场景3：全部汽车随机时间随机快慢充
# Sum_Flu = 0
# Sum_nonAC = 0
# Sum_InfUn = 0
# for l in range(1000):
#     user_power_SME = Ori_Curve_SME
#     EV_charge_power = blanklist
#
#     for k in range(5):  # 五台电动汽车
#         for i in range(5):  # 五个工作日白天
#             c = random.randint(0, 8)  # 随机0-8， 0超级，1357家用快充，2468慢充
#             if c in [2,4,6,8]:
#                 EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                     EV_slow_Fundamental,
#                                                                     random.randint(16, 18) + i * 48)  # 引入电动汽车后的一周内总功率曲线
#             if c in [1,3,5,7]:
#                 EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                     EV_fast_Fundamental_Family,
#                                                                     random.randint(16, 30) + i * 48)
#             if c == 0:
#                 EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                     EV_fast_Fundamental,
#                                                                     random.randint(18, 36) + i * 48)
#
#     if l == 1:
#         plt.figure()
#         plt.plot(Ori_Curve_SME, color='grey', label='原负荷曲线')  # 原负荷曲线
#         plt.plot(user_power_SME, color='tomato', label='引入电动汽车总负荷曲线')  # 引入电动汽车总负荷曲线
#         plt.plot(EV_charge_power, color='cornflowerblue', label='电动汽车充电负荷曲线')  # 电动汽车负荷曲线
#         plt.title('SME场景3：全部汽车随机时间快慢充')
#         plt.legend()
#         plt.xlabel('时间')
#         plt.ylabel('负荷')
#
#         # 创建DataFrame
#         data = {'Ori_Curve_SME': Ori_Curve_SME, 'user_power_SME': user_power_SME, 'EV_charge_power': EV_charge_power}
#         df = pd.DataFrame(data)
#
#         # 保存为Excel文件
#         df.to_excel('3.xlsx', index=False)
#
#     InPI_SME_Scenario, InPI_SME_Flu[3], InPI_SME_nonAC[3], InPI_SME_InfUn[3] = InPI(user_power_SME)
#
#     Sum_Flu = Sum_Flu + InPI_SME_Flu[3]
#     Sum_nonAC = Sum_nonAC + InPI_SME_nonAC[3]
#     Sum_InfUn = Sum_InfUn + InPI_SME_InfUn[3]
# InPI_SME_Flu[3] = Sum_Flu / 1000
# InPI_SME_nonAC[3] = Sum_nonAC / 1000
# InPI_SME_InfUn[3] = Sum_InfUn / 1000
#
#
# # 场景4：随机汽车数量随机时间随机快慢充
# Sum_Flu = 0
# Sum_nonAC = 0
# Sum_InfUn = 0
# for l in range(1000):
#     user_power_SME = Ori_Curve_SME
#     EV_charge_power = blanklist
#
#     for k in range(5):  # 五个工作日白天
#         EV_c = random.randint(0, 5)  # 每天随机充电汽车数量
#         for i in range(EV_c):  # 汽车数量
#             c = random.randint(0, 8)  # 随机0/1/2， 00超级，1357家用快充，2468慢充
#             if c in [2,4,6,8]:
#                 EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                     EV_slow_Fundamental,
#                                                                     random.randint(16, 18) + i * 48)  # 引入电动汽车后的一周内总功率曲线
#             if c in [1,3,5,7]:
#                 EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                     EV_fast_Fundamental_Family,
#                                                                     random.randint(16, 30) + i * 48)
#             if c == 0:
#                 EV_charge_power, user_power_SME = EV_Charging_Intro(user_power_SME, EV_charge_power,
#                                                                     EV_fast_Fundamental,
#                                                                     random.randint(18, 36) + i * 48)
#
#     if l == 1:
#         plt.figure()
#         plt.plot(Ori_Curve_SME, color='grey', label='原负荷曲线')  # 原负荷曲线
#         plt.plot(user_power_SME, color='tomato', label='引入电动汽车总负荷曲线')  # 引入电动汽车总负荷曲线
#         plt.plot(EV_charge_power, color='cornflowerblue', label='电动汽车充电负荷曲线')  # 电动汽车负荷曲线
#         plt.title('SME场景4：随机数量汽车随机时间随机快慢充')
#         plt.legend()
#         plt.xlabel('时间')
#         plt.ylabel('负荷')
#
#         # 创建DataFrame
#         data = {'Ori_Curve_SME': Ori_Curve_SME, 'user_power_SME': user_power_SME, 'EV_charge_power': EV_charge_power}
#         df = pd.DataFrame(data)
#
#         # 保存为Excel文件
#         df.to_excel('4.xlsx', index=False)
#
#     InPI_SME_Scenario, InPI_SME_Flu[4], InPI_SME_nonAC[4], InPI_SME_InfUn[4] = InPI(user_power_SME)
#
#     Sum_Flu = Sum_Flu + InPI_SME_Flu[4]
#     Sum_nonAC = Sum_nonAC + InPI_SME_nonAC[4]
#     Sum_InfUn = Sum_InfUn + InPI_SME_InfUn[4]
# InPI_SME_Flu[4] = Sum_Flu / 1000
# InPI_SME_nonAC[4] = Sum_nonAC / 1000
# InPI_SME_InfUn[4] = Sum_InfUn / 1000
#
#
# InPI_SME_Scenario, InPI_SME_Flu[0], InPI_SME_nonAC[0], InPI_SME_InfUn[0] = InPI(Ori_Curve_SME)  # 特别注意运算此InPI函数后，函数变量被归一化
#
#
# # 本征隐私性指标归一化
# # 求各指标值的最大值以便后续归一化（取极端条件即白噪声作为最值）
#
# # sum_Flu = 0
# # sum_nonAC = 0
# # sum_InfUn = 0
# # for i in range(1000):
# #     list = [(random.randint(0,1000) / 100) for k in range(336)]
# #     a, b, c, d = InPI(list)
# #     sum_Flu = sum_Flu + b
# #     sum_nonAC = sum_nonAC + c
# #     sum_InfUn = sum_InfUn + d
# #
# # max_InPI_Flu = sum_Flu / 1000
# # max_InPI_nonAC = 0.5
# # max_InPI_InfUn = sum_InfUn / 1000
#
# # ###由以上代码确定最终各指标值归一化最大最小值计算结果 max_InPI_Flu = 0.36; max_InPI_nonAC = 0.5; max_InPI_InfUn = 0.876
#
#
# max_InPI_Flu = 0.36
# min_InPI_Flu = 0
# for n in range(len(InPI_SME_Flu)):
#     InPI_SME_Flu[n] = (InPI_SME_Flu[n] - min_InPI_Flu) / (max_InPI_Flu - min_InPI_Flu)
#
# max_InPI_nonAC = 0.5
# min_InPI_nonAC = 0
# for n in range(len(InPI_SME_nonAC)):
#     InPI_SME_nonAC[n] = (InPI_SME_nonAC[n] - min_InPI_nonAC) / (max_InPI_nonAC - min_InPI_nonAC)
#
# max_InPI_InfUn = 0.876
# min_InPI_InfUn = 0
# for n in range(len(InPI_SME_InfUn)):
#     InPI_SME_InfUn[n] = (InPI_SME_InfUn[n] - min_InPI_InfUn) / (max_InPI_InfUn - min_InPI_InfUn)
#
#
# # 绘图观察本征隐私性变化
# x = [0, 1, 2, 3, 4]
# xvalues = ['未引入EV', '场景1', '场景2', '场景3', '场景4']
# # plt.figure()
# # plt.plot(InPI_SME_Flu)
# # plt.title('Fluctuation')
# # plt.xlabel('场景')
# # plt.xticks(x, xvalues)
# # plt.figure()
# # plt.plot(InPI_SME_nonAC)
# # plt.title('nonACor')
# # plt.xlabel('场景')
# # plt.xticks(x, xvalues)
# # plt.figure()
# # plt.plot(InPI_SME_InfUn)
# # plt.title('InfUncertainty')
# # plt.xlabel('场景')
# # plt.xticks(x, xvalues)
#
# # 计算InPI三合一指标变化
# for k in range(len(InPI_SME_3to1)):
#     InPI_SME_3to1[k] = (InPI_SME_Flu[k] + InPI_SME_nonAC[k] + InPI_SME_InfUn[k]) / 3
# print(InPI_SME_3to1)
# # plt.figure()
# # plt.plot(InPI_SME_3to1)
# # plt.title('三合一本征隐私性')
# # plt.xlabel('场景')
# # plt.ylabel('本征隐私性')
# # plt.xticks(x, xvalues)
# #
# plt.show()

