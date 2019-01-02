import random
import numpy as np
random.seed(1)


def get_d(A):
    d_l = np.sum(A,axis=1)
    d_d = np.sum(A,axis=0)
    # print(np.shape(A))
    # print(np.shape(d_l))
    # print(np.shape(d_d))
    return d_l,d_d


def lnc_sim_com_neighbors(A,d_l,d_d):
    lnc_length = len(A)
    lnc_sim = np.eye(lnc_length)

    for i in range(lnc_length):
        for j in range(lnc_length):
        # for j in range(i+1,lnc_length,1):
            if i==j:
                continue
            else:
                d_li = d_l[i]
                d_lj = d_l[j]
                S1L = np.sum(A[i] * A[j] / (d_d + np.spacing(1)))
                S1L = np.exp(-(S1L / (d_li * d_lj + np.spacing(1))))
                lnc_sim[i, j] = S1L

    return lnc_sim


def dis_sim_com_neighbors(A,d_l,d_d):
    dis_length = len(A[0])
    dis_sim = np.eye(dis_length)

    for i in range(dis_length):
        for j in range(dis_length):
            if i==j:
                continue
            else:
                d_di = d_d[i]
                d_dj = d_d[j]
                S1D = np.sum(A[:,i] * A[:,j] / (d_l + np.spacing(1)))
                S1D = np.exp(-(S1D / (d_di * d_dj + np.spacing(1))))
                dis_sim[i,j] = S1D

    return dis_sim


def lnc_sim_simRank(A,SD1,d_l,d_d):
    lnc_length = len(A)
    dis_length = len(A[0])
    lnc_sim = np.eye(lnc_length)

    for i in range(lnc_length):
        for j in range(lnc_length):
            if i==j:
                continue
            else:
                d_li = d_l[i]
                d_lj = d_l[j]
                S2L = 0
                for p in range(dis_length):
                    S2L += ((A[i,p] * A[j] * SD1[p]) / (d_d[p]*d_d+np.spacing(1)))
                S2L = np.sum(S2L)
                S2L = np.exp(-(S2L / (d_li * d_lj + np.spacing(1))))
                lnc_sim[i,j] = S2L

    return lnc_sim


def dis_sim_simRank(A,SL1,d_l,d_d):
    lnc_length = len(A)
    dis_length = len(A[0])
    dis_sim = np.eye(dis_length)

    for i in range(dis_length):
        for j in range(dis_length):
            if i==j:
                continue
            else:
                d_di = d_d[i]
                d_dj = d_d[j]
                S2L = 0
                for p in range(lnc_length):
                    S2L+=((A[p,i] * A[:,j] * SL1[p])/(d_l[p]*d_l+np.spacing(1)))
                S2L = np.sum(S2L)
                S2L = np.exp(-(S2L / (d_di * d_dj + np.spacing(1))))
                dis_sim[i,j] = S2L

    return dis_sim


def get_R(A,SL1,SD1,SL2,SD2,alpha):
    SL = SL1 * SL2
    SD = SD1 * SD2
    R1 = np.matmul(SL,A)
    R2 = np.matmul(A,SD)

    R = alpha * R1 + (1-alpha) * R2
    return R


def aNewMethodLDAP(A,alpha):
    d_l, d_d = get_d(A)
    SL1 = lnc_sim_com_neighbors(A, d_l, d_d)
    SD1 = dis_sim_com_neighbors(A, d_l, d_d)
    SL2 = lnc_sim_simRank(A,SD1,d_l,d_d)
    SD2 = dis_sim_simRank(A, SL1, d_l, d_d)
    R = get_R(A, SL1, SD1, SL2, SD2, alpha)

    return R




if __name__ == '__main__':
    alpha = 0.6
    A = np.loadtxt("../data/data_create/data/lnc_dis_association.txt")
    R = aNewMethodLDAP(A,alpha)
    print(R)