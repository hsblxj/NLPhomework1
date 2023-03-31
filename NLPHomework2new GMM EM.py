import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import math


def gaussian(X, mu_k, cov_k):
    norm = multivariate_normal(mean=mu_k, cov=cov_k)
    return norm.pdf(X)


if __name__ == '__main__':
    # 读文件
    d = pd.read_csv('height_data.csv', header=None)
    d = d.to_numpy()
    d = np.delete(d, 0)

    # 赋初值
    alpha1 = alpha2 = 0.5  # 男女比
    mu1 = 169.7  # 男生平均身高
    mu2 = 158.0  # 女生平均身高
    cov_1 = cov_2 = 1.0  # 方差
    precision = 0.00001  # 精度

    # E-M Step
    while True:
        # E
        Responsiveness_1 = np.zeros((len(d), 1))
        for i in range(len(d)):
            temp = alpha1 * gaussian(d[i], mu1, cov_1)/(alpha1 * gaussian(d[i], mu1, cov_1) +
                                                        alpha2 * gaussian(d[i], mu2, cov_2))
            Responsiveness_1[i] = temp
        Responsiveness_2 = np.zeros((len(d), 1))
        for i in range(len(d)):
            temp = alpha2 * gaussian(d[i], mu2, cov_2)/(alpha1 * gaussian(d[i], mu1, cov_1) +
                                                        alpha2 * gaussian(d[i], mu2, cov_2))
            Responsiveness_2[i] = temp

        # M
        temp1_1 = temp2_1 = temp1_2 = temp2_2 = temp1_3 = temp2_3 = 0.0
        for i in range(len(d)):
            temp1_1 += Responsiveness_1[i]*float(d[i])
            temp2_1 += Responsiveness_2[i]*float(d[i])
            temp1_2 += Responsiveness_1[i]
            temp2_2 += Responsiveness_2[i]
            temp1_3 += Responsiveness_1[i]*(float(d[i])-mu1)*(float(d[i])-mu1)
            temp2_3 += Responsiveness_2[i]*(float(d[i])-mu2)*(float(d[i])-mu2)
        if abs(mu1-temp1_1/temp1_2) < precision and \
            abs(mu2-temp2_1/temp2_2) < precision and \
            abs(cov_1-temp1_3/temp1_2) < precision and \
            abs(cov_2-temp2_3/temp2_2) < precision and \
            abs(alpha1-temp1_2/len(d)) < precision and \
                abs(alpha1 - temp1_2 / len(d)) < precision:
            break
        mu1 = temp1_1/temp1_2
        mu2 = temp2_1/temp2_2
        cov_1 = temp1_3/temp1_2
        cov_2 = temp2_3/temp2_2
        alpha1 = temp1_2/len(d)
        alpha2 = temp2_2/len(d)

    # 输出结果
    print("Estimated alpha: ", alpha1, alpha2)
    print("Estimated mu: ", mu1, mu2)
    print("Estimated cov: ", cov_1, cov_2)
