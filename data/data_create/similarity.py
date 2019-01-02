#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author:Yee
import numpy as np
##########################################################计算相似性

#读取文件lncRNA相似性矩阵
lnc_sim = np.loadtxt("data/lnc_sim.txt")
#读取disease相似性矩阵
dis_sim = np.loadtxt("data/dis_sim_matrix_process.txt")
#读取lncRNA-disease关联矩阵
lnc_dis = np.loadtxt("data/lnc_dis_association.txt")

def LncSim(l1,l2):                                   #输入的lnc-dis是两行
    row_sum1=[]
    all_sum=[]
    row_sum2=[]
    l1_position=np.argwhere(l1==1)                    #计算第一行为1的坐标
    l2_position=np.argwhere(l2==1)                    #计算第二行为1的坐标
    if (len(l1_position) == 0 or len(l2_position) == 0):   #如果其中一行lnc没有与dis关联时，则相似性为0
        row_lnc_sim = 0
    else:

        for i in range(len(l1_position)):
            for j in range(len(l2_position)):                     #这个时候dis_sim必须是矩阵
                a=dis_sim[[l1_position[i]],[l2_position[j]]]      #找出对应列的值
                row_sum1.append(a)                                #

            row_max=np.max(row_sum1)                              #找出对应列值得大小
            row_sum1.clear()                                      #清除每次循环，不然会一直叠加输入
            all_sum.append(row_max)                               #将求出每个L中的d的最大值保存

        for i in range(len(l2_position)):
            for j in range(len(l1_position)):
                b=dis_sim[[l2_position[i]],[l1_position[j]]]
                row_sum2.append(b)

            row_max=np.max(row_sum2)
            row_sum2.clear()
            all_sum.append(row_max)

        alll_sum=np.sum(all_sum)
        row_lnc_sim=alll_sum/(len(l1_position)+len(l2_position))
    return row_lnc_sim

#l1_l2=LncSim(lnc_dis[0],lnc_dis[1])
#print(l1_l2)

lnc_sim=np.zeros(shape=(lnc_dis.shape[0],lnc_dis.shape[0]))   #定义一个空的矩阵
for i in range(lnc_dis.shape[0]):
    print(lnc_dis[i])
    for j in range(lnc_dis.shape[0]):
        lnc_sim[i][j]=LncSim(lnc_dis[i],lnc_dis[j])            #将每个计算的值保存在矩阵对应的位置
np.savetxt("data/lnc_sim",lnc_sim)


#print(lnc_sim.shape)