# only support 1024 2048
import math
import numpy as np
import os
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
def PadToPowOfTwo(size):
    return pow(2, math.ceil(math.log2(size)))
# tensor core步骤之前最先准备的序列长度
# 1. 16*8/16*16 这样一次tensor core merge就结束？
# 2. 16 tensor core merge之后还要再一次cuda core merge？

# 实验目的
# 1. 理解这个过程中的memory同步需求  能够不加barrier吗？
# 2. 实际体验公式推导
# 3. 如何优化内存访问和计算  tensor core用于计算8*16的复数计算 效率？

# 实验步骤
# 1. 实现长度为16的子序列fft，然后实现16radix merging 验证结果
# 2. 把vertical+逆运算等也实现
# 3. 分析效率 和 访存 给出优化方案
# 4. 实现优化方案 放到gpu上

def ComplexAdd(input0, input1):
    # input0 [2]  input1 [2]

    return input0 + input1

def ComplexSub(input0, input1):
    # input0 [2]  input1 [2]

    return input0 - input1
def ComplexMul(input0, input1):
    return np.array([input0[0]*input1[0]-input0[1]*input1[1], input0[0]*input1[1]+input0[1]*input1[0]], dtype=np.float64)
def Radix2FFT(input):
    result = np.zeros(input.shape, dtype=np.float64)

    i= 0
    Twiddle0 = np.array([math.cos(-2.*math.pi*i/(2.)), math.sin(-2.*math.pi*i/(2.))], dtype=np.float64)
    Twiddle1 = np.array([math.cos(-2.*math.pi*(i+1)/(2.)), math.sin(-2.*math.pi*(i+1)/(2.))], dtype=np.float64)
    result[i] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle0))
    result[i + 1] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle1))# TODO: USE BUTTERFLY
    return result

def Radix4FFT(input):
    input0 = input[::2] # odd
    input1 = input[1::2] # even
    result0 = Radix2FFT(input0)
    result1 = Radix2FFT(input1)
    result = np.zeros(input.shape, dtype=np.float64)

    for i in range(2):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(4.)), math.sin(-2.*math.pi*i/(4.))], dtype=np.float64)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 2] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix8FFT(input):
    input0 = input[::2] # 
    input1 = input[1::2] # 
    result0 = Radix4FFT(input0)
    result1 = Radix4FFT(input1)
    
    result = np.zeros(input.shape, dtype=np.float64)
    for i in range(4):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(8.)), math.sin(-2.*math.pi*i/(8.))], dtype=np.float64)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 4] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix16FFT(input):
    # input is [16,2]

    input0 = input[::2] # even
    input1 = input[1::2] # odd
    result0 = Radix8FFT(input0)# even
    result1 = Radix8FFT(input1)# odd

    result = np.zeros(input.shape, dtype=np.float64)
    for i in range(8):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(16.)), math.sin(-2.*math.pi*i/(16.))], dtype=np.float64)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 8] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result

def Radix2FFT_inverse(input):
    result = np.zeros(input.shape, dtype=np.float64)

    i= 0
    Twiddle0 = np.array([math.cos(2.*math.pi*i/(2.)), math.sin(2.*math.pi*i/(2.))], dtype=np.float64)
    Twiddle1 = np.array([math.cos(2.*math.pi*(i+1)/(2.)), math.sin(2.*math.pi*(i+1)/(2.))], dtype=np.float64)
    result[i] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle0))
    result[i + 1] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle1))# TODO: USE BUTTERFLY
    return result

def Radix4FFT_inverse(input):
    input0 = input[::2] # odd
    input1 = input[1::2] # even
    result0 = Radix2FFT_inverse(input0)
    result1 = Radix2FFT_inverse(input1)
    result = np.zeros(input.shape, dtype=np.float64)

    for i in range(2):
        Twiddle = np.array([math.cos(2.*math.pi*i/(4.)), math.sin(2.*math.pi*i/(4.))], dtype=np.float64)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 2] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix8FFT_inverse(input):
    input0 = input[::2] # 
    input1 = input[1::2] # 
    result0 = Radix4FFT_inverse(input0)
    result1 = Radix4FFT_inverse(input1)
    
    result = np.zeros(input.shape, dtype=np.float64)
    for i in range(4):
        Twiddle = np.array([math.cos(2.*math.pi*i/(8.)), math.sin(2.*math.pi*i/(8.))], dtype=np.float64)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 4] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix16FFT_inverse(input):
    # input is [16,2]

    input0 = input[::2] # even
    input1 = input[1::2] # odd
    result0 = Radix8FFT_inverse(input0)# even
    result1 = Radix8FFT_inverse(input1)# odd

    result = np.zeros(input.shape, dtype=np.float64)
    for i in range(8):
        Twiddle = np.array([math.cos(2.*math.pi*i/(16.)), math.sin(2.*math.pi*i/(16.))], dtype=np.float64)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 8] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result


def DFT16(input):
    # input is [16,2]

    result = np.zeros(input.shape, dtype=np.float64)
    for i in range(16):
        result[i, 0] = 0
        result[i, 1] = 0
        for j in range(16):
            temp = ComplexMul(input[j], np.array([math.cos(-2.*math.pi*i*j/(16.)), math.sin(-2.*math.pi*i*j/(16.))], dtype=np.float64))
            result[i, 0] += temp[0]
            result[i, 1] += temp[1]
    return result

def FFT(paddedwidth, img_row):# w, c

    img_width = img_row.shape[0]
    output_frequency_row = np.zeros((paddedwidth+2, 4), dtype=np.float64)

    sub_array_num = paddedwidth // 16 // 16

    mergeradix16_rg_inputs = []
    mergeradix16_ba_inputs = []

    sub_sub_array_num = paddedwidth // 16
    for i in range(sub_sub_array_num):
        input = np.zeros((16, 2), dtype=np.float64)
        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=np.float64)
            else:
                input[j] = img_row[pixelIndex, 0:2]
        mergeradix16_rg_inputs.append(Radix16FFT(input))

        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=np.float64)
                # if horizontal:
                #     input[:,1] = 1.# set alpha to 1
            else:
                input[j] = img_row[pixelIndex, 2:4]
        mergeradix16_ba_inputs.append(Radix16FFT(input))

    T_real = np.zeros((16, 16), dtype=np.float64)
    T_imag = np.zeros((16, 16), dtype=np.float64)

    F_real = np.zeros((16, 16), dtype=np.float64)
    F_imag = np.zeros((16, 16), dtype=np.float64)

    for i in range(16):
        for j in range(16):
            if i == 0:
                T_real[i,j] = 1
                T_imag[i,j] = 0
                F_real[i,j] = 1
                F_imag[i,j] = 0
            else:
                T_real[i,j] = math.cos(-2.*math.pi*i*j/(16.*16.))
                T_imag[i,j] = math.sin(-2.*math.pi*i*j/(16.*16.))
                F_real[i,j] = math.cos(-2.*math.pi*i*j/(16.))
                F_imag[i,j] = math.sin(-2.*math.pi*i*j/(16.))

    sub_array_list = []
    for i in range(sub_array_num):
        X_in_rg_real = np.zeros((16, 16), dtype=np.float64)
        X_in_rg_imag = np.zeros((16, 16), dtype=np.float64)
        X_in_ba_real = np.zeros((16, 16), dtype=np.float64)
        X_in_ba_imag = np.zeros((16, 16), dtype=np.float64)

        # fill data
        for index_row in range(16):
            X_in_rg_real[index_row, :] = mergeradix16_rg_inputs[sub_array_num * index_row + i][:,0]
            X_in_rg_imag[index_row, :] = mergeradix16_rg_inputs[sub_array_num * index_row + i][:,1]
            X_in_ba_real[index_row, :] = mergeradix16_ba_inputs[sub_array_num * index_row + i][:,0]
            X_in_ba_imag[index_row, :] = mergeradix16_ba_inputs[sub_array_num * index_row + i][:,1]


        TxX_in_rg_real = T_real * X_in_rg_real - T_imag * X_in_rg_imag
        TxX_in_rg_imag = T_imag * X_in_rg_real + T_real * X_in_rg_imag

        TxX_in_ba_real = T_real * X_in_ba_real - T_imag * X_in_ba_imag
        TxX_in_ba_imag = T_imag * X_in_ba_real + T_real * X_in_ba_imag

        Aa_rg = np.matmul(F_real, TxX_in_rg_real)
        Bb_rg = np.matmul(F_imag, TxX_in_rg_imag)
        Ba_rg = np.matmul(F_imag, TxX_in_rg_real)
        Ab_rg = np.matmul(F_real, TxX_in_rg_imag)

        Aa_ba = np.matmul(F_real, TxX_in_ba_real)
        Bb_ba = np.matmul(F_imag, TxX_in_ba_imag)
        Ba_ba = np.matmul(F_imag, TxX_in_ba_real)
        Ab_ba = np.matmul(F_real, TxX_in_ba_imag)

        sub_array_list.append([Aa_rg - Bb_rg, Ba_rg + Ab_rg, Aa_ba - Bb_ba, Ba_ba + Ab_ba]) #https://blog.csdn.net/weixin_39274659/article/details/109629569

    if sub_array_num == 1:
        return sub_array_list[0][0].flatten(), sub_array_list[0][1].flatten(), sub_array_list[0][2].flatten(), sub_array_list[0][3].flatten()

    # merge 256 length signal to 1024/2048 length
    # radix 4/8 merge
    # if sub_array_num == 4 or sub_array_num == 8:
    else:
        X_in_rg_real = np.zeros((sub_array_num, 256), dtype=np.float64)
        X_in_rg_imag = np.zeros((sub_array_num, 256), dtype=np.float64)
        X_in_ba_real = np.zeros((sub_array_num, 256), dtype=np.float64)
        X_in_ba_imag = np.zeros((sub_array_num, 256), dtype=np.float64)

        T_real = np.zeros((sub_array_num, 256), dtype=np.float64)
        T_imag = np.zeros((sub_array_num, 256), dtype=np.float64)

        F_real = np.zeros((sub_array_num, sub_array_num), dtype=np.float64)
        F_imag = np.zeros((sub_array_num, sub_array_num), dtype=np.float64)

        # fill T and F N1==sub_array_num N2==16X16
        for i in range(sub_array_num):
            for j in range(256):
                T_real[i,j] = math.cos(-2.*math.pi*i*j/(sub_array_num*256.))
                T_imag[i,j] = math.sin(-2.*math.pi*i*j/(sub_array_num*256.))
        for i in range(sub_array_num):
            for j in range(sub_array_num):       
                F_real[i,j] = math.cos(-2.*math.pi*i*j/(sub_array_num))
                F_imag[i,j] = math.sin(-2.*math.pi*i*j/(sub_array_num))

        # fill data
        for index_row in range(sub_array_num):
            X_in_rg_real[index_row, :] = sub_array_list[index_row][0].flatten()
            X_in_rg_imag[index_row, :] = sub_array_list[index_row][1].flatten()
            X_in_ba_real[index_row, :] = sub_array_list[index_row][2].flatten()
            X_in_ba_imag[index_row, :] = sub_array_list[index_row][3].flatten()

        TxX_in_rg_real = T_real * X_in_rg_real - T_imag * X_in_rg_imag
        TxX_in_rg_imag = T_imag * X_in_rg_real + T_real * X_in_rg_imag

        TxX_in_ba_real = T_real * X_in_ba_real - T_imag * X_in_ba_imag
        TxX_in_ba_imag = T_imag * X_in_ba_real + T_real * X_in_ba_imag

        Aa_rg = np.matmul(F_real, TxX_in_rg_real)
        Bb_rg = np.matmul(F_imag, TxX_in_rg_imag)
        Ba_rg = np.matmul(F_imag, TxX_in_rg_real)
        Ab_rg = np.matmul(F_real, TxX_in_rg_imag)

        Aa_ba = np.matmul(F_real, TxX_in_ba_real)
        Bb_ba = np.matmul(F_imag, TxX_in_ba_imag)
        Ba_ba = np.matmul(F_imag, TxX_in_ba_real)
        Ab_ba = np.matmul(F_real, TxX_in_ba_imag)

        # rg
        rg_real = (Aa_rg - Bb_rg).flatten()
        rg_imag = (Ba_rg + Ab_rg).flatten()
        
        # for i in range(paddedwidth // 2 + 1):# r0 to r[N/2], g[N/2+1]... g[N-1] g[0] g[N/2]
        #     if i == 0:
        #         output_frequency_row[i][0:2] = np.array([ # r[0]
        #             rg_real[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth][0:2] = np.array([ #g[0] 
        #             rg_imag[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth + 1][0:2] = np.array([ #g[N/2]
        #             -0.5*rg_imag[paddedwidth // 2]-0.5*rg_imag[paddedwidth // 2], 0.5*rg_real[paddedwidth // 2]-0.5*rg_real[paddedwidth // 2]# 0#TODO:
        #             ], dtype=np.float64)
        #     else:
        #         output_frequency_row[i][0:2] = np.array([
        #             0.5*rg_real[i]+0.5*rg_real[paddedwidth-i], 0.5*rg_imag[paddedwidth-i] - 0.5*rg_imag[i],
        #             ], dtype=np.float64)
                
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][0:2] = np.array([
        #             0.5*rg_imag[i + paddedwidth // 2 + 1]+0.5*rg_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*rg_real[i + paddedwidth // 2 + 1]-0.5*rg_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=np.float64)# TODO:
        # ba
        ba_real = (Aa_ba - Bb_ba).flatten()
        ba_imag = (Ba_ba + Ab_ba).flatten()
        # for i in range(paddedwidth // 2 + 1):
        #     if i == 0:
        #         output_frequency_row[i][2:4] = np.array([ # r[0]
        #             ba_real[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth][2:4] = np.array([ #g[0]
        #             ba_imag[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth + 1][2:4] = np.array([ #g[N/2]
        #             -0.5*ba_imag[paddedwidth // 2]-0.5*ba_imag[paddedwidth // 2], 0.5*ba_real[paddedwidth // 2]-0.5*ba_real[paddedwidth // 2]# 0
        #             ], dtype=np.float64)

        #     else:
        #         output_frequency_row[i][2:4] = np.array([
        #             0.5*ba_real[i]+0.5*ba_real[paddedwidth-i], 0.5*ba_imag[paddedwidth-i]-0.5*ba_imag[i],
        #             ], dtype=np.float64)
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][2:4] = np.array([
        #             0.5*ba_imag[i + paddedwidth // 2 + 1]+0.5*ba_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*ba_real[i + paddedwidth // 2 + 1]-0.5*ba_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=np.float64)

        # return output_frequency_row

        return rg_real, rg_imag, ba_real, ba_imag
    # else:
    #     assert(False)
    #     return []

def FFT_inverse(paddedwidth, img_row):# w, c

    img_width = img_row.shape[0]
    output_frequency_row = np.zeros((paddedwidth+2, 4), dtype=np.float64)

    sub_array_num = paddedwidth // 16 // 16

    mergeradix16_rg_inputs = []
    mergeradix16_ba_inputs = []

    sub_sub_array_num = paddedwidth // 16
    for i in range(sub_sub_array_num):
        input = np.zeros((16, 2), dtype=np.float64)
        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=np.float64)
            else:
                input[j] = img_row[pixelIndex, 0:2]
        mergeradix16_rg_inputs.append(Radix16FFT_inverse(input))

        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=np.float64)
                # if horizontal:
                #     input[:,1] = 1.# set alpha to 1
            else:
                input[j] = img_row[pixelIndex, 2:4]
        mergeradix16_ba_inputs.append(Radix16FFT_inverse(input))

    T_real = np.zeros((16, 16), dtype=np.float64)
    T_imag = np.zeros((16, 16), dtype=np.float64)

    F_real = np.zeros((16, 16), dtype=np.float64)
    F_imag = np.zeros((16, 16), dtype=np.float64)

    for i in range(16):
        for j in range(16):
            if i == 0:
                T_real[i,j] = 1
                T_imag[i,j] = 0
                F_real[i,j] = 1
                F_imag[i,j] = 0
            else:
                T_real[i,j] = math.cos(2.*math.pi*i*j/(16.*16.))
                T_imag[i,j] = math.sin(2.*math.pi*i*j/(16.*16.))
                F_real[i,j] = math.cos(2.*math.pi*i*j/(16.))
                F_imag[i,j] = math.sin(2.*math.pi*i*j/(16.))

    sub_array_list = []
    for i in range(sub_array_num):
        X_in_rg_real = np.zeros((16, 16), dtype=np.float64)
        X_in_rg_imag = np.zeros((16, 16), dtype=np.float64)
        X_in_ba_real = np.zeros((16, 16), dtype=np.float64)
        X_in_ba_imag = np.zeros((16, 16), dtype=np.float64)

        # fill data
        for index_row in range(16):
            X_in_rg_real[index_row, :] = mergeradix16_rg_inputs[sub_array_num * index_row + i][:,0]
            X_in_rg_imag[index_row, :] = mergeradix16_rg_inputs[sub_array_num * index_row + i][:,1]
            X_in_ba_real[index_row, :] = mergeradix16_ba_inputs[sub_array_num * index_row + i][:,0]
            X_in_ba_imag[index_row, :] = mergeradix16_ba_inputs[sub_array_num * index_row + i][:,1]


        TxX_in_rg_real = T_real * X_in_rg_real - T_imag * X_in_rg_imag
        TxX_in_rg_imag = T_imag * X_in_rg_real + T_real * X_in_rg_imag

        TxX_in_ba_real = T_real * X_in_ba_real - T_imag * X_in_ba_imag
        TxX_in_ba_imag = T_imag * X_in_ba_real + T_real * X_in_ba_imag

        Aa_rg = np.matmul(F_real, TxX_in_rg_real)
        Bb_rg = np.matmul(F_imag, TxX_in_rg_imag)
        Ba_rg = np.matmul(F_imag, TxX_in_rg_real)
        Ab_rg = np.matmul(F_real, TxX_in_rg_imag)

        Aa_ba = np.matmul(F_real, TxX_in_ba_real)
        Bb_ba = np.matmul(F_imag, TxX_in_ba_imag)
        Ba_ba = np.matmul(F_imag, TxX_in_ba_real)
        Ab_ba = np.matmul(F_real, TxX_in_ba_imag)

        sub_array_list.append([Aa_rg - Bb_rg, Ba_rg + Ab_rg, Aa_ba - Bb_ba, Ba_ba + Ab_ba]) #https://blog.csdn.net/weixin_39274659/article/details/109629569

    if sub_array_num == 1:
        return sub_array_list[0][0].flatten(), sub_array_list[0][1].flatten(), sub_array_list[0][2].flatten(), sub_array_list[0][3].flatten()

    # merge 256 length signal to 1024/2048 length
    # radix 4/8 merge
    # if sub_array_num == 4 or sub_array_num == 8:
    else:
        X_in_rg_real = np.zeros((sub_array_num, 256), dtype=np.float64)
        X_in_rg_imag = np.zeros((sub_array_num, 256), dtype=np.float64)
        X_in_ba_real = np.zeros((sub_array_num, 256), dtype=np.float64)
        X_in_ba_imag = np.zeros((sub_array_num, 256), dtype=np.float64)

        T_real = np.zeros((sub_array_num, 256), dtype=np.float64)
        T_imag = np.zeros((sub_array_num, 256), dtype=np.float64)

        F_real = np.zeros((sub_array_num, sub_array_num), dtype=np.float64)
        F_imag = np.zeros((sub_array_num, sub_array_num), dtype=np.float64)

        # fill T and F N1==sub_array_num N2==16X16
        for i in range(sub_array_num):
            for j in range(256):
                T_real[i,j] = math.cos(2.*math.pi*i*j/(sub_array_num*256.))
                T_imag[i,j] = math.sin(2.*math.pi*i*j/(sub_array_num*256.))
        for i in range(sub_array_num):
            for j in range(sub_array_num):       
                F_real[i,j] = math.cos(2.*math.pi*i*j/(sub_array_num))
                F_imag[i,j] = math.sin(2.*math.pi*i*j/(sub_array_num))

        # fill data
        for index_row in range(sub_array_num):
            X_in_rg_real[index_row, :] = sub_array_list[index_row][0].flatten()
            X_in_rg_imag[index_row, :] = sub_array_list[index_row][1].flatten()
            X_in_ba_real[index_row, :] = sub_array_list[index_row][2].flatten()
            X_in_ba_imag[index_row, :] = sub_array_list[index_row][3].flatten()

        TxX_in_rg_real = T_real * X_in_rg_real - T_imag * X_in_rg_imag
        TxX_in_rg_imag = T_imag * X_in_rg_real + T_real * X_in_rg_imag

        TxX_in_ba_real = T_real * X_in_ba_real - T_imag * X_in_ba_imag
        TxX_in_ba_imag = T_imag * X_in_ba_real + T_real * X_in_ba_imag

        Aa_rg = np.matmul(F_real, TxX_in_rg_real)
        Bb_rg = np.matmul(F_imag, TxX_in_rg_imag)
        Ba_rg = np.matmul(F_imag, TxX_in_rg_real)
        Ab_rg = np.matmul(F_real, TxX_in_rg_imag)

        Aa_ba = np.matmul(F_real, TxX_in_ba_real)
        Bb_ba = np.matmul(F_imag, TxX_in_ba_imag)
        Ba_ba = np.matmul(F_imag, TxX_in_ba_real)
        Ab_ba = np.matmul(F_real, TxX_in_ba_imag)

        # rg
        rg_real = (Aa_rg - Bb_rg).flatten()
        rg_imag = (Ba_rg + Ab_rg).flatten()
        
        # for i in range(paddedwidth // 2 + 1):# r0 to r[N/2], g[N/2+1]... g[N-1] g[0] g[N/2]
        #     if i == 0:
        #         output_frequency_row[i][0:2] = np.array([ # r[0]
        #             rg_real[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth][0:2] = np.array([ #g[0] 
        #             rg_imag[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth + 1][0:2] = np.array([ #g[N/2]
        #             -0.5*rg_imag[paddedwidth // 2]-0.5*rg_imag[paddedwidth // 2], 0.5*rg_real[paddedwidth // 2]-0.5*rg_real[paddedwidth // 2]# 0#TODO:
        #             ], dtype=np.float64)
        #     else:
        #         output_frequency_row[i][0:2] = np.array([
        #             0.5*rg_real[i]+0.5*rg_real[paddedwidth-i], 0.5*rg_imag[paddedwidth-i] - 0.5*rg_imag[i],
        #             ], dtype=np.float64)
                
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][0:2] = np.array([
        #             0.5*rg_imag[i + paddedwidth // 2 + 1]+0.5*rg_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*rg_real[i + paddedwidth // 2 + 1]-0.5*rg_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=np.float64)# TODO:
        # ba
        ba_real = (Aa_ba - Bb_ba).flatten()
        ba_imag = (Ba_ba + Ab_ba).flatten()
        # for i in range(paddedwidth // 2 + 1):
        #     if i == 0:
        #         output_frequency_row[i][2:4] = np.array([ # r[0]
        #             ba_real[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth][2:4] = np.array([ #g[0]
        #             ba_imag[0], 0.
        #             ], dtype=np.float64)
        #         output_frequency_row[paddedwidth + 1][2:4] = np.array([ #g[N/2]
        #             -0.5*ba_imag[paddedwidth // 2]-0.5*ba_imag[paddedwidth // 2], 0.5*ba_real[paddedwidth // 2]-0.5*ba_real[paddedwidth // 2]# 0
        #             ], dtype=np.float64)

        #     else:
        #         output_frequency_row[i][2:4] = np.array([
        #             0.5*ba_real[i]+0.5*ba_real[paddedwidth-i], 0.5*ba_imag[paddedwidth-i]-0.5*ba_imag[i],
        #             ], dtype=np.float64)
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][2:4] = np.array([
        #             0.5*ba_imag[i + paddedwidth // 2 + 1]+0.5*ba_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*ba_real[i + paddedwidth // 2 + 1]-0.5*ba_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=np.float64)

        # return output_frequency_row

        return rg_real, rg_imag, ba_real, ba_imag
def HorizontalFFTAndSplitOneForTwo(img):#
    h, w, _ = img.shape
    paddedWidth = PadToPowOfTwo(w)

    output_img = np.zeros((h, paddedWidth+2, 4), dtype=np.float64)
    
    for i in tqdm(range(h)):
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedWidth, img[i,:])

        # frequency_row = 
        # output_img[i,:] = frequency_row
    return output_img

    



def GetRelativeError(ours, ref):
    error_matrix = ours.copy()

    error_matrix[ref != 0.] = np.abs(ours[ref != 0.] - ref[ref != 0.]) / np.abs(ref[ref != 0.])
    error_matrix[ref == 0.] = np.abs(ours[ref == 0.] - ref[ref == 0.])
    error = np.average(error_matrix)
    return error

def HorizontalFFTErrorUnitTest():
    input_img = cv2.imread("img/input_1280x720.hdr", cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=np.float64)
    rgb_channel4[:,:,:3] = rgb_img

    h, w, _ = rgb_channel4.shape

    paddedWidth = PadToPowOfTwo(w)

    output_img = np.zeros((h, paddedWidth+2, 4), dtype=np.float64)
    for i in tqdm(range(h)):
        paddedInput = np.zeros((paddedWidth, 4), dtype=np.float64)
        paddedInput[:w] = rgb_channel4[i,:]
        # paddedInput[:,3] = 1.# set alpha to 1
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedWidth, rgb_channel4[i,:])
        ref_rg = np.fft.fft(paddedInput[...,0] + 1j * paddedInput[...,1])
        ref_ba = np.fft.fft(paddedInput[...,2] + 1j * paddedInput[...,3])
        
        error0 = GetRelativeError(rg_real, np.real(ref_rg))
        error1 = GetRelativeError(rg_imag, np.imag(ref_rg))
        error2 = GetRelativeError(ba_real, np.real(ref_ba))
        error3 = GetRelativeError(ba_imag, np.imag(ref_ba))
        
        epsilon = 1e-10
        if error0 > epsilon:
            print(i)
            print(error0)
        if error1 > epsilon:
            print(i)
            print(error1)
        if error2 > epsilon:
            print(i)
            print(error2)
        if error3 > epsilon:
            print(i)
            print(error3)

def TwoDimenFFTUnitTest():
    input_img = cv2.imread("img/input_1280x720.hdr", cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=np.float64)
    rgb_channel4[:,:,:3] = rgb_img

    h, w, _ = rgb_channel4.shape

    paddedWidth = PadToPowOfTwo(w)
    paddedHeight = PadToPowOfTwo(h)

    output_img = np.zeros((h, paddedWidth, 4), dtype=np.float64)
    
    for i in tqdm(range(h)):
        paddedInput = np.zeros((paddedWidth, 4), dtype=np.float64)
        paddedInput[:w] = rgb_channel4[i,:]
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedWidth, rgb_channel4[i,:])

        output_img[i,:,0] = rg_real
        output_img[i,:,1] = rg_imag
        output_img[i,:,2] = ba_real
        output_img[i,:,3] = ba_imag


    output_fft_img = np.zeros((paddedHeight, paddedWidth, 4), dtype=np.float64)
    
    for j in tqdm(range(paddedWidth)):
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedHeight, output_img[:,j])

        output_fft_img[:,j,0] = rg_real
        output_fft_img[:,j,1] = rg_imag
        output_fft_img[:,j,2] = ba_real
        output_fft_img[:,j,3] = ba_imag

    paddedInput2D = np.zeros((paddedHeight, paddedWidth, 4), dtype=np.float64)
    paddedInput2D[:h,:w,:] = rgb_channel4
    ref_fft_rg = np.fft.fft2(paddedInput2D[:,:,0] + 1j * paddedInput2D[:,:,1])
    ref_fft_ba = np.fft.fft2(paddedInput2D[:,:,2] + 1j * paddedInput2D[:,:,3])
    output_fft_ref = np.zeros(output_fft_img.shape, dtype=np.float64)
    output_fft_ref[:,:,0] = np.real(ref_fft_rg)
    output_fft_ref[:,:,1] = np.imag(ref_fft_rg)
    output_fft_ref[:,:,2] = np.real(ref_fft_ba)
    output_fft_ref[:,:,3] = np.imag(ref_fft_ba)

    relative_error = np.zeros(output_fft_img.shape, dtype=np.float64)
    relative_error[output_fft_ref == 0.] = np.abs(output_fft_img[output_fft_ref == 0.])
    relative_error[output_fft_ref != 0.] = np.abs(output_fft_img[output_fft_ref != 0.] - output_fft_ref[output_fft_ref != 0.]) / np.abs(output_fft_ref[output_fft_ref != 0.])
    

    print(np.average(relative_error))
def HorizontalFFTTwoForOne_UnitTest():
    input_img = cv2.imread("img/input_1280x720.hdr", cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=np.float64)
    rgb_channel4[:,:,:3] = rgb_img
    horizontal_output = HorizontalFFTAndSplitOneForTwo(rgb_channel4)

    converted = horizontal_output.copy()
    converted[:,:,0] = horizontal_output[:,:,2]
    converted[:,:,2] = horizontal_output[:,:,0]
    cv2.imwrite("horizontal_output.exr", converted)
if __name__ == "__main__":
    # test: random points fft
    # random_array = np.random.random((1024, 4))

    # rg_real, rg_imag, ba_real, ba_imag = FFT(1024, random_array)

    # rg_ref = np.fft.fft(random_array[...,0] + 1j * random_array[...,1])
    # ba_ref = np.fft.fft(random_array[...,2] + 1j * random_array[...,3])

    # print(rg_real)
    # print(rg_imag)
    # print(rg_ref)
    # print(ba_real)
    # print(ba_imag)
    # print(ba_ref)

    # test: random points fft and inverse fft

    # random_array = np.random.random((2048, 4))
    # paddedLength = PadToPowOfTwo(random_array.shape[0])
    # rg_real, rg_imag, ba_real, ba_imag = FFT(paddedLength, random_array)
    # fft_result = np.zeros((paddedLength, 4), dtype=np.float64)
    # fft_result[:,0] = rg_real
    # fft_result[:,1] = rg_imag
    # fft_result[:,2] = ba_real
    # fft_result[:,3] = ba_imag
    # rg_real, rg_imag, ba_real, ba_imag = FFT_inverse(paddedLength, fft_result)

    # fft_result[:,0] = rg_real * (1. / paddedLength)
    # fft_result[:,1] = rg_imag * (1. / paddedLength)
    # fft_result[:,2] = ba_real * (1. / paddedLength)
    # fft_result[:,3] = ba_imag * (1. / paddedLength)

    # errormetrices = np.abs(fft_result)
    # errormetrices[random_array != 0.] = np.abs(random_array[random_array != 0.] - fft_result[random_array != 0.]) / np.abs(random_array[random_array != 0.])
    # print(np.average(errormetrices))

    # test1: do fft on image rows
    # HorizontalFFTErrorUnitTest()

    # test2: fft two for one (one complex fft for two real series)
    HorizontalFFTTwoForOne_UnitTest()

    # test3: vertical fft 
    # TwoDimenFFTUnitTest()
    # test4: vertical fft and multiply

    # test5: inverse vertical fft
    
    # test6: inverse horizontal fft