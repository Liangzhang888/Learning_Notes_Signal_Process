#!/usr/bin/env python
# coding: utf-8
'''
程序说明：
    1.时频分析方法装在get_AF()函数里，并从中调用其他需要的函数
    2.给出了HHT理论的实现方法
    3.需要计算振荡参数的话，直接调用get_oscillation_parameter函数即可
    导入的模块有些可能用不上，这是因为这个程序是在总程序基础上删改得到的
'''
import os
import math
import cmath
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import solve
from scipy import signal
from scipy import fftpack
from scipy import interpolate
from scipy import optimize
from scipy.optimize import fsolve
#HHT理论相关程序
    #获取极值索引，默认忽略端点
def extreme_index(data):
    max_index=[]
    min_index=[]
    dx=np.zeros(len(data))
    dx[:-1]=np.diff(data)
    dx[-1]=dx[-2]
    dx=np.sign(dx)
    d2x=np.zeros(len(data))
    d2x[1:]=np.diff(dx)
    d2x[0]=d2x[1]
    for i in np.arange(1,len(d2x)-1):
        if d2x[i]==-2:
            max_index.append(i)
        if d2x[i]==2:
            min_index.append(i)                 
    return max_index,min_index

    #包络拟合,控制参数包括端点处理方法、插值方法
def get_envelope(data,border_type='OMIT',interpolation='cubic'):
    imf={'x':data['x'],'y_pri':data['y'],'y_new':data['y']}
    if interpolation=='cubic':
        if border_type=='OMIT':
            max_index,min_index=extreme_index(data['y'])
            if len(max_index)>3 and len(min_index)>3:                                      #这样设置，包含了多种OMIT方法下的情形，如递减、极大值与极小值点均唯一等情况。这里数字需要自己设定（考虑到边界条件，此值应大于3）。
                cubic_max = interpolate.interp1d(np.array(data['x'])[max_index],np.array(data['y'])[max_index],kind='cubic') 
                cubic_min = interpolate.interp1d(np.array(data['x'])[min_index],np.array(data['y'])[min_index],kind='cubic')
                data_x_temp_start=data['x'][max_index[0]:] if max_index[0]>min_index[0] else data['x'][min_index[0]:]
                data_x_temp_end=data['x'][:max_index[-1]+1] if max_index[-1]<min_index[-1] else data['x'][:min_index[-1]+1]
                data_x_temp=list(set(data_x_temp_start).intersection(set(data_x_temp_end)))#取交集
                data_x_temp.sort()                                                         #排序，保证x序列递增
                envelope_max = cubic_max(data_x_temp)                                      #得到的数据类型为ndarray
                envelope_min = cubic_min(data_x_temp) 
                envelope_ave=0.5*(envelope_max+envelope_min)
                #data_y_temp_start=data['y'][max_index[0]:] if max_index[0]>min_index[0] else data['y'][min_index[0]:]
                #data_y_temp_end=data['y'][:max_index[-1]+1] if max_index[-1]<min_index[-1] else data['y'][:min_index[-1]+1]
                #data_y_temp=list(set(data_y_temp_start).intersection(set(data_y_temp_end))) x坐标因为互不相同，可以通过取交集来避开索引问题，这里对y就不适用了
                t=pd.DataFrame({'x':data['x'],'y':data['y']})                              #构造字典，根据得到的data_x_temp对data['y']进行更新
                temp=t[t['x']>=data_x_temp[0]]
                final=temp[temp['x']<=data_x_temp[len(data_x_temp)-1]]
                data_y_temp=final['y']  
                imf_temp=data_y_temp-envelope_ave
                imf['x']=np.array(data_x_temp)
                imf['y_pri']=data_y_temp
                imf['y_new']=imf_temp
                return imf
            else:
                return 1
        if border_type=='Extreme_estimation':#针对单一信号
            max_index,min_index=extreme_index(data['y'])
            if len(max_index)>2 and len(min_index)>2:#保证供极值估值的数据点大于2
                time_max_y=list(data['x'][max_index])
                time_max_x=list(range(2,len(max_index)+2))
                value_max_y=list(data['y'][max_index])
                value_max_x=list(range(2,len(max_index)+2))
                time_min_y=list(data['x'][min_index])
                time_min_x=list(range(2,len(min_index)+2))
                value_min_y=list(data['y'][min_index])
                value_min_x=list(range(2,len(min_index)+2))
                poly_power_max=len(max_index)-1 if len(max_index)-1<=100 else 100 #多项式阶数最高限制在100
                poly_power_min=len(min_index)-1 if len(min_index)-1<=100 else 100
                poly_time_max={} #构造字典，键值与多项式次数保持一致,储存多项式系数
                poly_value_max={}
                poly_time_min={}
                poly_value_min={}
                for i in range(poly_power_max+1): #对极大值出现时间与数值进行多项式拟合，多项式最大次数取决于极大值点个数。极小值的处理方式同理。
                    func_time_max=np.polyfit(time_max_x,time_max_y,poly_power_max-i) #func为ndarray对象
                    poly_time_max[poly_power_max-i]=func_time_max
                    func_value_max=np.polyfit(value_max_x,value_max_y,poly_power_max-i) 
                    poly_value_max[poly_power_max-i]=func_value_max
                for i in range(poly_power_min+1):
                    func_time_min=np.polyfit(time_min_x,time_min_y,poly_power_min-i) #func为ndarray对象
                    poly_time_min[poly_power_min-i]=func_time_min
                    func_value_min=np.polyfit(value_min_x,value_min_y,poly_power_min-i) 
                    poly_value_min[poly_power_min-i]=func_value_min
                error_time_max={}
                error_value_max={}
                error_time_min={}
                error_value_min={}
                for i in range(poly_power_max+1):  #计算所有次幂多项式极大值拟合数据平方误差，极小值数据处理同理。
                    error_time_single=[]
                    error_value_single=[]
                    for j in range(len(max_index)): #计算所有数据点数据
                        data_time=[]
                        for k in range(poly_power_max-i+1): #计算特定多项式下特定数据点拟合误差,k表示多项式系数索引
                            data_time.append(poly_time_max[poly_power_max-i][k]*(time_max_x[j]**(poly_power_max-k)))
                        error_time_single.append((time_max_y[j]-sum(data_time))**2)
                    for j in range(len(max_index)): 
                        data_value=[]
                        for k in range(poly_power_max-i+1):
                            data_value.append(poly_value_max[poly_power_max-i][k]*(value_max_x[j]**(poly_power_max-k)))
                        error_value_single.append((value_max_y[j]-sum(data_value))**2)    
                    error_time_max[poly_power_max-i]=sum(error_time_single)
                    error_value_max[poly_power_max-i]=sum(error_value_single)
                for i in range(poly_power_min+1):  
                    error_time_single=[]
                    error_value_single=[]
                    for j in range(len(min_index)): 
                        data_time=[]
                        for k in range(poly_power_min-i+1): 
                            data_time.append(poly_time_min[poly_power_min-i][k]*(time_min_x[j]**(poly_power_min-k)))
                        error_time_single.append((time_min_y[j]-sum(data_time))**2)
                    for j in range(len(min_index)): 
                        data_value=[]
                        for k in range(poly_power_min-i+1):
                            data_value.append(poly_value_min[poly_power_min-i][k]*(value_min_x[j]**(poly_power_min-k)))
                        error_value_single.append((value_min_y[j]-sum(data_value))**2)    
                    error_time_min[poly_power_min-i]=sum(error_time_single)
                    error_value_min[poly_power_min-i]=sum(error_value_single)
                #筛选最优多项式,获取其次幂信息
                poly_time_max_best=0     
                poly_value_max_best=0
                poly_time_min_best=0   
                poly_value_min_best=0 
                for i in range(1,len(error_time_max)):
                    if error_time_max[i]<=error_time_max[poly_time_max_best]:
                        poly_time_max_best=i     
                for i in range(1,len(error_value_max)):
                    if error_value_max[i]<=error_value_max[poly_value_max_best]:
                        poly_value_max_best=i
                for i in range(1,len(error_time_min)):
                    if error_time_min[i]<=error_time_min[poly_time_min_best]:
                        poly_time_min_best=i                     
                for i in range(1,len(error_value_min)):
                    if error_value_min[i]<=error_value_min[poly_value_min_best]:
                        poly_value_min_best=i
                #估测端点（包括头、尾两侧）极值数据
                data_temp=[] 
                if poly_time_max_best==0:
                    data_temp.append(poly_time_max[0][0])    
                else :
                    for i in range(poly_time_max_best+1): 
                        data_temp.append(poly_time_max[poly_time_max_best][i]*1)
                time_max_top_estimation=sum(data_temp)
                data_temp=[] 
                if poly_time_max_best==0:
                    data_temp.append(poly_time_max[0][0])    
                else :
                    for i in range(poly_time_max_best+1): 
                        data_temp.append(poly_time_max[poly_time_max_best][i]*((len(max_index)+2)**(poly_time_max_best-i)))
                time_max_end_estimation=sum(data_temp)
                data_temp=[] 
                if poly_value_max_best==0:
                    data_temp.append(poly_value_max[0][0])    
                else :
                    for i in range(poly_value_max_best+1): 
                        data_temp.append(poly_value_max[poly_value_max_best][i]*1)
                value_max_top_estimation=sum(data_temp)
                data_temp=[] 
                if poly_value_max_best==0:
                    data_temp.append(poly_value_max[0][0])    
                else :
                    for i in range(poly_value_max_best+1): 
                        data_temp.append(poly_value_max[poly_value_max_best][i]*((len(max_index)+2)**(poly_value_max_best-i)))
                value_max_end_estimation=sum(data_temp)
                data_temp=[] 
                if poly_time_min_best==0:
                    data_temp.append(poly_time_min[0][0])    
                else :
                    for i in range(poly_time_min_best+1): 
                        data_temp.append(poly_time_min[poly_time_min_best][i]*1)
                time_min_top_estimation=sum(data_temp)
                data_temp=[] 
                if poly_time_min_best==0:
                    data_temp.append(poly_time_min[0][0])    
                else :
                    for i in range(poly_time_min_best+1): 
                        data_temp.append(poly_time_min[poly_time_min_best][i]*((len(min_index)+2)**(poly_time_min_best-i)))
                time_min_end_estimation=sum(data_temp)
                data_temp=[] 
                if poly_value_min_best==0:
                    data_temp.append(poly_value_min[0][0])    
                else :
                    for i in range(poly_value_min_best+1): 
                        data_temp.append(poly_value_min[poly_value_min_best][i]*1)
                value_min_top_estimation=sum(data_temp) 
                data_temp=[] 
                if poly_value_min_best==0:
                    data_temp.append(poly_value_min[0][0])    
                else :
                    for i in range(poly_value_min_best+1): 
                        data_temp.append(poly_value_min[poly_value_min_best][i]*((len(min_index)+2)**(poly_value_min_best-i)))
                value_min_end_estimation=sum(data_temp) 
                #极值数据扩充，得到包含端点的拟合数据
                #当曲线左侧极值出现时间拟合结果大于采样数据时间起点时，令其为data['x'][0]，同时令该点取值为极值拟合结果
                #当曲线右边侧极值出现时间拟合结果小于采样数据时间起点时，令其为data['x'][-1]，同时令该点取值为极值拟合结果
                if time_max_top_estimation>=data['x'][0]:
                    time_max_top_estimation=data['x'][0]
                if time_max_end_estimation<=data['x'][-1]:
                    time_max_end_estimation=data['x'][-1]
                if time_min_top_estimation>=data['x'][0]:
                    time_min_top_estimation=data['x'][0]
                if time_min_end_estimation<=data['x'][-1]:
                    time_min_end_estimation=data['x'][-1]
                data_max_x=[time_max_top_estimation]
                data_max_x.extend(time_max_y)
                data_max_x.extend([time_max_end_estimation])
                data_max_y=[value_max_top_estimation]
                data_max_y.extend(value_max_y)
                data_max_y.extend([value_max_end_estimation])
                data_min_x=[time_min_top_estimation]
                data_min_x.extend(time_min_y)
                data_min_x.extend([time_min_end_estimation])
                data_min_y=[value_min_top_estimation]
                data_min_y.extend(value_min_y)
                data_min_y.extend([value_min_end_estimation])
                cubic_max = interpolate.interp1d(np.array(data_max_x),np.array(data_max_y),kind='cubic') 
                cubic_min = interpolate.interp1d(np.array(data_min_x),np.array(data_min_y),kind='cubic')                                                   
                envelope_max = cubic_max(list(data['x']))  
                envelope_min = cubic_min(list(data['x'])) 
                envelope_ave=0.5*(envelope_max+envelope_min) 
                imf_temp=data['y']-envelope_ave
                imf['x']=data['x']
                imf['y_pri']=data['y']
                imf['y_new']=imf_temp
                return imf
            else:
                return 1

    #获取特定IMF分量,使用sift_mode参数控制筛选模式，并给出端点处理与插值方法设定的入口。
def get_IMF(data,sift_mode='Cauthy',border_type='OMIT',interpolation='cubic',imf_num=2,liner_sift_num=4,emd_update_num=1):
    data_temp={}
    data_new={}
    if sift_mode=='Cauthy':                          #类柯西准奏判定方法
        threshold=0.8                                #Huang参考值0.2-0.3，根据信号具体情况进行相应修改（很有必要！！）
        thre=2*threshold
        data_temp=data
        while thre>threshold:                        #获取第一个IMF分量，sift_mode进行控制.循环次数若过多，很容易导致极值点个数太少，无法求包络。需要修改相关函数
            imf=get_envelope(data_temp,border_type=border_type,interpolation=interpolation)
            if imf==1:                               #极点个数不满足要求的情形
                return data_temp,1                   #返回上一个筛选（sift_mode）过程的IMF与特征反馈信号
            else:
                top=sum((imf['y_pri']-imf['y_new'])**2)
                bottom=sum((imf['y_pri'])**2)
                data_temp={'x':imf['x'],'y':imf['y_new']}
                data_new={'x':imf['x'],'y':imf['y_pri']-imf['y_new']}
                thre=top/bottom
    if sift_mode=='Fixed':                           #以固定次数作为筛选终止条件，仍需添加IMF极值点数目限制
        data_temp=data
        while imf_num>0: 
            imf=get_envelope(data_temp,border_type=border_type,interpolation=interpolation)
            if imf==1:                               
                return data_temp,1                 
            else:               
                data_temp={'x':imf['x'],'y':imf['y_new']}
                data_new={'x':imf['x'],'y':imf['y_pri']-imf['y_new']}
                imf_num-=1
    if sift_mode=='Liner_reduction_slow':            #采用线性递减（慢速）的形式对筛选次数进行控制
        data_temp=data
        sift_num=liner_sift_num-emd_update_num+1     #最大次数需要依靠经验设定，函数给出设定入口
        if sift_num<1:                               #保证至少筛选一次
            sift_num=1
        while sift_num>0: 
            imf=get_envelope(data_temp,border_type=border_type,interpolation=interpolation)
            if imf==1:                              
                return data_temp,1                
            else:               
                data_temp={'x':imf['x'],'y':imf['y_new']}
                data_new={'x':imf['x'],'y':imf['y_pri']-imf['y_new']}
                sift_num-=1
    if sift_mode=='Liner_reduction_fast':            #采用线性递减（快速）的形式对筛选次数进行控制
        data_temp=data
        sift_num=liner_sift_num-2*emd_update_num+2   
        if sift_num<1:  
            sift_num=1
        while sift_num>0: 
            imf=get_envelope(data_temp,border_type=border_type,interpolation=interpolation)
            if imf==1:                             
                return data_temp,1                
            else:               
                data_temp={'x':imf['x'],'y':imf['y_new']}
                data_new={'x':imf['x'],'y':imf['y_pri']-imf['y_new']}
                sift_num-=1
    if sift_mode=='Liner_addition_slow':             #采用线性递增（慢速）的形式对筛选次数进行控制
        data_temp=data
        sift_num=liner_sift_num+emd_update_num-1  
        while sift_num>0: 
            imf=get_envelope(data_temp,border_type=border_type,interpolation=interpolation)
            if imf==1:                               
                return data_temp,1                   
            else:               
                data_temp={'x':imf['x'],'y':imf['y_new']}
                data_new={'x':imf['x'],'y':imf['y_pri']-imf['y_new']}
                sift_num-=1
    if sift_mode=='Liner_addition_fast':             #采用线性递增（快速）的形式对筛选次数进行控制
        data_temp=data
        sift_num=liner_sift_num+2*emd_update_num-2   
        while sift_num>0: 
            imf=get_envelope(data_temp,border_type=border_type,interpolation=interpolation)
            if imf==1:                            
                return data_temp,1                 
            else:               
                data_temp={'x':imf['x'],'y':imf['y_new']}
                data_new={'x':imf['x'],'y':imf['y_pri']-imf['y_new']}
                sift_num-=1
    return data_temp,data_new

    #对信号进行EMD分解,由depose_mode控制分解终止条件，并给出筛选模式、端点处理与插值方法的设置入口。函数返回IMF个数以及IMF数据（字典形式）。
def get_EMD(data,depose_mode='Monotonic',sift_mode='Cauthy',border_type='OMIT',interpolation='cubic',emd_num=5,imf_num=2,liner_sift_num=4):
    num=0
    imf_set={}
    if depose_mode=='Monotonic':                 #以IMF极值点数目作为分解终止条件（数学上具有一般性），包含了Huang的单调函数情形。
        mono=0
        data_temp=data
        while True:
            num+=1
            imf_temp,data_new=get_IMF(data_temp,sift_mode=sift_mode,border_type=border_type,interpolation=interpolation,imf_num=imf_num,liner_sift_num=liner_sift_num,emd_update_num=num)
            imf_set[num]={'x'+str(num):imf_temp['x'],'y'+str(num):imf_temp['y']}
            if data_new==1:                      #达到EMD终止条件，退出循环
                break
            else:
                data_temp=data_new
    if depose_mode=='Fixed':                     #以固定次数作为分解终止条件，仍需添加IMF极值点数目限制
        data_temp=data                           
        if emd_num==1:
            imf_temp,data_new=get_IMF(data_temp,sift_mode=sift_mode,border_type=border_type,interpolation=interpolation,imf_num=imf_num,liner_sift_num=liner_sift_num,emd_update_num=num)
            imf_set[1]={'x1':imf_temp['x'],'y1':imf_temp['y']}
        else:
            while emd_num>0:
                num+=1
                imf_temp,data_new=get_IMF(data_temp,sift_mode=sift_mode,border_type=border_type,interpolation=interpolation,imf_num=imf_num,liner_sift_num=liner_sift_num,emd_update_num=num)
                if emd_num>1:#3.9 EMD分解次数固定时（emd_num初始值大于1），最后一个IMF由原数据减去前列IMF得到。如果不需要这样处理，注释掉这部分判断语句即可
                    imf_set[num]={'x'+str(num):imf_temp['x'],'y'+str(num):imf_temp['y']}
                else: 
                    imf_set[num]={'x'+str(num):data_temp['x'],'y'+str(num):data_temp['y']}
                if data_new==1:                      #达到EMD终止条件，退出循环
                    break
                else:
                    data_temp=data_new
                    emd_num-=1
    return num,imf_set    

#获取单一信号希尔伯特变换结果
def get_HT_FFT(imf_x,imf_y):
    amplitude=[]
    phase=[]
    ht_y=-fftpack.hilbert(imf_y)                       #此处Hilbert变换结果中，基于sin函数的移项角为-pi/2
    for num in range(len(imf_y)):
        com_y=complex(imf_y[num],ht_y[num])            
        amplitude.append(abs(com_y))                   #幅值
        phase.append(cmath.phase(com_y)+0.5*np.pi)     #相位。对Hilbert变换得到的角度进行修正，包括：初相角，角度范围变为[-pi/2,3pi/2]                                          
    dimf_y=np.zeros(len(imf_y))
    dimf_y[:-1]=np.diff(phase)
    if dimf_y[0]<0:                                    #注意到相位周期性变化的特征，考虑相位差分量突变的情形（正频率下突变量为负数），在突变处令差分量为前一时刻值（这种办法会造成一定误差，但不影响整体）。
        dimf_y[0]=dimf_y[1]                            #第一个点为负数的话，令其等于第二个值
    for check in range(1,len(dimf_y)-1):
        if dimf_y[check]<0:
            dimf_y[check]=dimf_y[check-1]        
    dimf_y[-1]=dimf_y[-2]
    dimf_x=np.zeros(len(imf_x))
    dimf_x[:-1]=np.diff(imf_x)
    dimf_x[-1]=dimf_x[-2]
    omega=dimf_y/dimf_x                                #频率（为简化程序，设末尾数据点频率与上一个点相同。潜在问题：相位边界点可能存在异常数据）
    return amplitude,phase,omega

#获取EMD所得IMF集合的希尔伯特变换结果
def get_HT_FFT_set(imf_set):
    AF_set={}
    for i in range(len(imf_set)):                          #通过len(imf_set)获取imf数量，之前的函数给出了包含imf数量信息的返回值，是为了方便查看该信息，但这不是必须的
        imf_x=imf_set[i+1]['x'+str(i+1)]                   
        imf_y=np.array(list(imf_set[i+1]['y'+str(i+1)]))   #过程中转换为list以消除array原索引
        dimf_x=np.zeros(len(imf_x))
        dimf_x[:-1]=np.diff(imf_x)
        dimf_x[-1]=dimf_x[-2]
        amplitude,phase,omega=get_HT_FFT(imf_x,imf_y)
        real_omega=np.zeros(len(omega))
        real_omega[0]=omega[0]
        for j in range(1,len(real_omega)):
            real_omega[j]=(omega[j-1]-real_omega[j-1])*dimf_x[j]/imf_x[j-1]+real_omega[j-1]
        #根据情况选择omega或是real_omega
        AF_set[i+1]={'x'+str(i+1):imf_set[i+1]['x'+str(i+1)],'amplitude'+str(i+1):amplitude,'phase'+str(i+1):phase,'omega'+str(i+1):omega} 
    return AF_set

#经验AM-FM方法实现
def get_Maximum_estimation(imf_x,imf_y,max_index,min_index,border_type):
   #出于简化程序考虑，这里未考虑考虑极值点为空的情形。若对于幅值波动很小的信号进行分析时程序报错，将解调部分程序注释掉即可
    max_index.extend(min_index)#合并极值索引
    max_index.sort()
    extreme_indexes=max_index
    extreme_value_x=np.array(list(imf_x))[extreme_indexes]
    extreme_value_y=np.array(list(imf_y))[extreme_indexes]
    for j in range(len(extreme_value_y)):
        if extreme_value_y[j]<0:
            extreme_value_y[j]=-extreme_value_y[j]    
    if border_type=='OMIT':
        extreme_cubic = interpolate.interp1d(extreme_value_x,extreme_value_y,kind='cubic')
        temp_x=np.array(list(imf_x))[extreme_indexes[0]:extreme_indexes[-1]+1]
        temp_y=np.array(list(imf_y))[extreme_indexes[0]:extreme_indexes[-1]+1]
        extreme_envelope = extreme_cubic(temp_x)
        for k in range(len(temp_y)):
            temp_y[k]=temp_y[k]/extreme_envelope[k]
    if border_type=='Extreme_estimation':#这种情形下的极值估计，与EMD环节中有所不同，主要体现在这里将极大值与极小值同一处理为极（大）值
        max_index=extreme_value_x 
        time_max_y=list(extreme_value_x)
        time_max_x=list(range(2,len(max_index)+2))
        value_max_y=list(extreme_value_y)
        value_max_x=list(range(2,len(max_index)+2))
        poly_power_max=len(max_index)-1 if len(max_index)-1<=100 else 100 #多项式阶数最高限制在100
        poly_time_max={} #构造字典，键值与多项式次数保持一致,储存多项式系数
        poly_value_max={}
        for k in range(len(max_index)): #对极大值出现时间与数值进行多项式拟合，多项式最大次数取决于极大值点个数。极小值的处理方式同理。
            func_time_max=np.polyfit(time_max_x,time_max_y,poly_power_max-k) #func为ndarray对象
            poly_time_max[poly_power_max-k]=func_time_max
            func_value_max=np.polyfit(value_max_x,value_max_y,poly_power_max-k) 
            poly_value_max[poly_power_max-k]=func_value_max
        error_time_max={}
        error_value_max={}
        for k in range(len(max_index)):  #计算所有次幂多项式极大值拟合数据平方误差，极小值数据处理同理。
            error_time_single=[]
            error_value_single=[]
            for h in range(len(max_index)): #计算所有数据点数据
                data_time=[]
                for p in range(poly_power_max-k+1): #计算特定多项式下特定数据点拟合误差,j表示多项式系数索引
                    data_time.append(poly_time_max[poly_power_max-k][p]*(time_max_x[h]**(poly_power_max-p)))
                error_time_single.append((time_max_y[h]-sum(data_time))**2)
            for h in range(len(max_index)): 
                data_value=[]
                for p in range(poly_power_max-k+1):
                    data_value.append(poly_value_max[poly_power_max-k][p]*(value_max_x[h]**(poly_power_max-p)))
                error_value_single.append((value_max_y[h]-sum(data_value))**2)    
            error_time_max[poly_power_max-k]=sum(error_time_single)
            error_value_max[poly_power_max-k]=sum(error_value_single)
        #筛选最优多项式,获取其次幂信息
        poly_time_max_best=0     
        poly_value_max_best=0
        for k in range(1,len(error_time_max)):
            if error_time_max[k]<=error_time_max[poly_time_max_best]:
                poly_time_max_best=k     
        for k in range(1,len(error_value_max)):
            if error_value_max[k]<=error_value_max[poly_value_max_best]:
                poly_value_max_best=k
        #估测极值数据
        data_temp=[] 
        if poly_time_max_best==0:
            data_temp.append(poly_time_max[0][0])    
        else :
            for k in range(poly_time_max_best+1): 
                data_temp.append(poly_time_max[poly_time_max_best][k]*1)
        time_max_top_estimation=sum(data_temp)
        data_temp=[] 
        if poly_time_max_best==0:
            data_temp.append(poly_time_max[0][0])    
        else :
            for k in range(poly_time_max_best+1): 
                data_temp.append(poly_time_max[poly_time_max_best][k]*((len(max_index)+2)**(poly_time_max_best-k)))
        time_max_end_estimation=sum(data_temp)
        data_temp=[] 
        if poly_value_max_best==0:
            data_temp.append(poly_value_max[0][0])    
        else :
            for k in range(poly_value_max_best+1): 
                data_temp.append(poly_value_max[poly_value_max_best][k]*1)
        value_max_top_estimation=sum(data_temp)
        data_temp=[] 
        if poly_value_max_best==0:
            data_temp.append(poly_value_max[0][0])    
        else :
            for k in range(poly_value_max_best+1): 
                data_temp.append(poly_value_max[poly_value_max_best][k]*((len(max_index)+2)**(poly_value_max_best-k)))
        value_max_end_estimation=sum(data_temp)
        #极值数据扩充，得到包含端点的拟合数据
        if time_max_top_estimation>=imf_x[0]:
            time_max_top_estimation=imf_x[0]
        if time_max_end_estimation<=imf_x[-1]:
            time_max_end_estimation=imf_x[-1]
        data_max_x=[time_max_top_estimation]
        data_max_x.extend(extreme_value_x)
        data_max_x.extend([time_max_end_estimation])
        data_max_y=[value_max_top_estimation]
        data_max_y.extend(extreme_value_y)
        data_max_y.extend([value_max_end_estimation]) 
        extreme_cubic = interpolate.interp1d(np.array(data_max_x),np.array(data_max_y),kind='cubic')
        temp_x=np.array(list(imf_x))
        temp_y=np.array(list(imf_y))
        extreme_envelope = extreme_cubic(temp_x)
        for k in range(len(temp_y)):
            temp_y[k]=temp_y[k]/extreme_envelope[k] 
    #AM-FM(NHT方法具体体现)
    phase=[]
    imf_x=temp_x                   
    imf_y=temp_y   #过程中转换为list以消除array原索引
    ht_y=-fftpack.hilbert(imf_y)                       #此处Hilbert变换结果中，基于sin函数的移项角为-pi/2
    for num in range(len(imf_y)):
        com_y=complex(imf_y[num],ht_y[num])            
        phase.append(cmath.phase(com_y)+0.5*np.pi)     #相位。对Hilbert变换得到的角度进行修正，包括：初相角，角度范围变为[-pi/2,3pi/2]                                          
    dimf_y=np.zeros(len(imf_y))
    dimf_y[:-1]=np.diff(phase)
    if dimf_y[0]<0:                                    #注意到相位周期性变化的特征，考虑相位差分量突变的情形（正频率下突变量为负数），在突变处令差分量为前一时刻值（这种办法会造成一定误差，但不影响整体）。
        dimf_y[0]=dimf_y[1]                            #第一个点为负数的话，令其等于第二个值
    for check in range(1,len(dimf_y)-1):
        if dimf_y[check]<0:
            dimf_y[check]=dimf_y[check-1]        
    dimf_y[-1]=dimf_y[-2]
    dimf_x=np.zeros(len(imf_x))
    dimf_x[:-1]=np.diff(imf_x)
    dimf_x[-1]=dimf_x[-2]
    omega=dimf_y/dimf_x 
    return temp_x,temp_y,extreme_envelope,omega,phase

#Direct方法-获取瞬时相位
def get_Phase(temp_y):
    if np.max(abs(temp_y))>1:
        base=np.max(abs(temp_y)) 
    else:
        base=1          
    imf_y=temp_y
    phase=list(np.zeros(len(imf_y)))
    dimf_y=np.zeros(len(imf_y))
    dimf_y[:-1]=np.diff(imf_y)
    dimf_y[-1]=dimf_y[-2]
    for num in range(len(imf_y)):                  #直接获取相位，取值范围[0,2pi]
        sin_value =imf_y[num]/base if abs(imf_y[num]/base)<=1 else (imf_y[num]/base)/abs(imf_y[num]/base) #尽管已经对base取了imf_y最大值，频率数据还是有异常情况出现，系统提示反三角函数运算错误，因此这里再次进行限定,限定后系统未报错，但没有改善频率异常数据
        if imf_y[num]==0:
            if dimf_y[num]>0:
                phase[num]=0
            else:
                phase[num]=np.pi
        elif imf_y[num]>0:
            if dimf_y[num]>0:
                phase[num]=np.arcsin(sin_value)
            else:
                phase[num]=np.pi-np.arcsin(sin_value)
        else:
            if dimf_y[num]<0:
                phase[num]=np.pi-np.arcsin(sin_value)
            else:
                phase[num]=2*np.pi+np.arcsin(sin_value) 
    return phase

#Direct方法-获取瞬时频率
def get_Direct(temp_x,temp_y,section_num):
    imf_x=temp_x
    phase=get_Phase(temp_y=temp_y)
    diff_phase=np.zeros(len(phase))
    diff_phase[:-1]=np.diff(phase)
    if diff_phase[0]<0:                                    
        diff_phase[0]=diff_phase[1]                            
    for check in range(1,len(diff_phase)-1):
        if diff_phase[check]<0:
            diff_phase[check]=diff_phase[check-1]        
    diff_phase[-1]=diff_phase[-2]
    dimf_x=np.zeros(len(imf_x))
    dimf_x[:-1]=np.diff(imf_x)
    dimf_x[-1]=dimf_x[-2]
    omega=diff_phase/dimf_x
    '''
    针对频率出现异常值的情形，对异常值进行判断和替换，亦适用于后续频率时变情形下的分析
    基本思想：根据数据长度，对信号进行分段（分段数量依赖于数据长度，即频率与采样时间，需要根据信号特点进行经验性设定），在每段进行异常值判定、替换
    经反复测试，可以发现，分段数量的值对处理后的频率结果有着显著影响，分段数量过大会导致明显的区间边界效应（这是由具有相对较大偏差的频率分量占比较高导致的）。
    经测试，可以发现，正弦调频时频率分析结果不对（采样时间较长时），原因分析：一方面，是EMD环节出了问题；另一方面，信号幅值不恒定，原有方法不适用
    '''
    section_num=section_num #分段数量
    exception_value=1.0#异常值筛选标准,需要根据信号形式进行调整，取1相当于将区间值用区间中位数代替，体现直线近似曲线的思想
    k=int(len(omega)/section_num) #向下取整
    for section in range(section_num+1):
        if section<section_num:
            sec_omega=np.array(list(omega[k*section:k*(section+1)]))
            median=np.median(sec_omega)
            if sec_omega[0]/median>exception_value or median/sec_omega[0]>exception_value:
                sec_omega[0]=median
            for j in range(1,len(sec_omega)):
                if sec_omega[j]/median>exception_value or median/sec_omega[j]>exception_value:
                    sec_omega[j]=sec_omega[j-1] #前项值填充 
            omega[k*section:k*(section+1)]=sec_omega
        else:
            sec_omega=np.array(list(omega[k*section:]))
            if len(sec_omega)>2:
                median=np.median(sec_omega)
                if sec_omega[0]/median>exception_value or median/sec_omega[0]>exception_value:
                    sec_omega[0]=median
                for j in range(1,len(sec_omega)):
                    if sec_omega[j]/median>exception_value or median/sec_omega[j]>exception_value:
                        sec_omega[j]=sec_omega[j-1] 
            omega[k*section:]=sec_omega
    return omega,phase
    
#定义幅频解析函数get_AF，获取IMF/单分量信号的幅值、频率、相位信息
def get_AF(imf_set,method='Amplitude_decomposition',demodulation_type='maximum_estimation',IFE_method='coupling',A_type='',F_type='',border_type='OMIT',distance=0,section_num=500,delta=0):
    AF_set={}
    if method=='Amplitude_decomposition':
        for i in range(len(imf_set)):
            max_index,min_index=extreme_index(np.array(list(imf_set[i+1]['y'+str(i+1)])))
            max_column=np.array(list(imf_set[i+1]['y'+str(i+1)]))[max_index]
            min_column=np.array(list(imf_set[i+1]['y'+str(i+1)]))[min_index]
#            max_column_update=max_column[~np.isnan(max_column)] #采样频率会影响极值的确定，在有些情况下会出现极值空值导致判断出错，因此需要先进行判断筛选。为简化分析调幅信号的程序，先不考虑这种特殊情况。
#            min_column_update=min_column[~np.isnan(min_column)]
            if len(max_column)+len(min_column)>2: #若极值点个数少于2，则Maximum_estimation与Logrithm_EMD方法失效，统一使用HT方法进行分析。这也是HT方法的优势所在。但在这种情况下，对于幅度调制信号来说，HT方法又是失效的。目前看来，幅度调制信号数据曲线极值点个数少于2时没有适用的分析方法。 
                if demodulation_type=='Maximum_estimation':
                    temp_x,temp_y,extreme_envelope,omega_new,phase_new=get_Maximum_estimation(imf_x=imf_set[i+1]['x'+str(i+1)],imf_y=imf_set[i+1]['y'+str(i+1)],max_index=max_index,min_index=min_index,border_type=border_type)       
                amplitude=extreme_envelope
                omega,phase=get_Direct(temp_x,temp_y,section_num=section_num)
                imf_x=temp_x
                dimf_x=np.zeros(len(imf_x))
                dimf_x[:-1]=np.diff(imf_x)
                dimf_x[-1]=dimf_x[-2]
                real_omega=np.zeros(len(omega))
                real_omega[0]=omega[0]
                for j in range(1,len(real_omega)):
                    real_omega[j]=(omega[j-1]-real_omega[j-1])*dimf_x[j]/imf_x[j-1]+real_omega[j-1]
                if demodulation_type=='IFEZF':
                    omega=OMEGA_new
                    real_omega=omega_new
                AF_set[i+1]={'x'+str(i+1):imf_x,'amplitude'+str(i+1):amplitude,'phase'+str(i+1):phase,'omega'+str(i+1):omega}
            else:    #当IMF为趋势线时，采用普通HT方法，这里采用基于FFT的HT算法
                imf_x=imf_set[i+1]['x'+str(i+1)]                   
                imf_y=np.array(list(imf_set[i+1]['y'+str(i+1)]))   
                amplitude,phase,omega=get_HT_FFT(imf_x,imf_y)
                dimf_x=np.zeros(len(imf_x))
                dimf_x[:-1]=np.diff(imf_x)
                dimf_x[-1]=dimf_x[-2]
                real_omega=np.zeros(len(omega))
                real_omega[0]=omega[0]
                for j in range(1,len(real_omega)):
                    real_omega[j]=(omega[j-1]-real_omega[j-1])*dimf_x[j]/imf_x[j-1]+real_omega[j-1]
                AF_set[i+1]={'x'+str(i+1):imf_x,'amplitude'+str(i+1):amplitude,'phase'+str(i+1):phase,'omega'+str(i+1):omega}
    return AF_set

#振荡参数求解函数：通过拟合，获取振荡参数：初始幅值，衰减系数，振荡频率
def get_oscillation_parameter(AF_set):  
    def amp(x,a,b): #调幅部分拟合
        return a*x+b
    def fre(x,c): #频率部分拟合
        return c*x**0
    oscillation_parameter_set={}
    for i in range(len(AF_set)):
        amplitude_x=list(AF_set[i+1]['x'+str(i+1)])
        amplitude_y=list(np.log(AF_set[i+1]['amplitude'+str(i+1)]))
        a,b=optimize.curve_fit(amp,amplitude_x,amplitude_y)[0]
        frequency_x=list(AF_set[i+1]['x'+str(i+1)])
        frequency_y=list(AF_set[i+1]['omega'+str(i+1)])
        c=optimize.curve_fit(fre,frequency_x,frequency_y)[0]
        oscillation_parameter_set[i+1]={'max_amplitude':np.exp(b),'coefficient':-a,'frequency':c}
    return oscillation_parameter_set

#仅适用于特定低频振荡信号IMF绘图
def plot_IMF_special(data,imf_num,imf_set,play_mode='multi'):
    if play_mode=='multi':
        plt.subplot(3,1,1)
        plt.plot(data['x'],data['y'],color='red',label='data')
        plt.legend(loc="upper right")
        plt.subplot(3,1,2)
        plt.plot(imf_set[1]['x1'],imf_set[1]['y1'],color='green',label='IMF1')
        plt.legend(loc="upper right")
        plt.subplot(3,1,3)
        plt.plot(imf_set[2]['x2'],imf_set[2]['y2'],color='blue',label='IMF2')
        plt.legend(loc="upper right")
        plt.xlabel('time')
        plt.show()
        plt.close()
    if play_mode=='single':
        plt.plot(data['x'],data['y'],label='data')
        for i in range(imf_num):
            plt.plot(imf_set[i+1]['x'+str(i+1)],imf_set[i+1]['y'+str(i+1)],label='IMF'+str(i+1))
        plt.legend(loc="best")
        plt.show()
        plt.close() 
    return

#绘制IMF分量幅值、频率、相位信息,为方便测验，增加参数imf_order来对特定IMF进行选择。
def plot_AF(AF_set,imf_order=0):
    if imf_order==0:                                       #默认情况下对所有IMF进行分解
        for i in range(len(AF_set)):
            plt.subplot(3,1,1)
            plt.plot(AF_set[i+1]['x'+str(i+1)],AF_set[i+1]['amplitude'+str(i+1)],color='red',label='IMF'+str(i+1)+'_amplitude')
            plt.legend(loc="upper right")
            plt.subplot(3,1,2)
            plt.plot(AF_set[i+1]['x'+str(i+1)],AF_set[i+1]['omega'+str(i+1)],color='green',label='IMF'+str(i+1)+'_frequency')
            plt.legend(loc="upper right")
            plt.subplot(3,1,3)
            plt.plot(AF_set[i+1]['x'+str(i+1)],AF_set[i+1]['phase'+str(i+1)],color='blue',label='IMF'+str(i+1)+'_phase')
            plt.legend(loc="upper right")
            plt.xlabel('time')
            plt.show()
            plt.close()
    else:
        plt.subplot(3,1,1)
        plt.scatter(AF_set[imf_order]['x'+str(imf_order)],AF_set[imf_order]['amplitude'+str(imf_order)],s=1,color='red',label='IMF'+str(imf_order)+'_amplitude')
        plt.legend(loc="upper right")
        plt.subplot(3,1,2)
        plt.scatter(AF_set[imf_order]['x'+str(imf_order)],AF_set[imf_order]['omega'+str(imf_order)],s=1,color='green',label='IMF'+str(imf_order)+'_frequency')
        plt.legend(loc="upper right")
        plt.subplot(3,1,3)
        plt.plot(AF_set[imf_order]['x'+str(imf_order)],AF_set[imf_order]['phase'+str(imf_order)],color='blue',label='IMF'+str(imf_order)+'_phase')
        plt.legend(loc="upper right")
        plt.show()
        plt.close()
    return

#低频振荡信号测试-经验AM-FM方法
fs=100000#采样频率10k
data_x=np.linspace(0.0,10.0,fs,endpoint=False)#允许最后一个点不被采样，以得到规整数据
func=lambda x:1.5*np.e**(-0.2*x)*np.cos(2*np.pi*1.5*x)+1.48*np.e**(-0.7*x)*np.cos(2*np.pi*0.5*x)
data_y=func(data_x)
data={'x':data_x,'y':data_y}
imf_num,imf_set=get_EMD(data,depose_mode='Fixed',sift_mode='Cauthy',border_type='OMIT',interpolation='cubic',emd_num=2)
plot_IMF_special(data,imf_num,imf_set,play_mode='multi')
AF_set_NHT1=get_AF(imf_set,method='Amplitude_decomposition',demodulation_type='Maximum_estimation',border_type='OMIT') 
for i in range(len(AF_set_NHT1)):#频率转换
    AF_set_NHT1[i+1]['omega'+str(i+1)]= AF_set_NHT1[i+1]['omega'+str(i+1)]/(2*np.pi)
plot_AF(AF_set_NHT1)


