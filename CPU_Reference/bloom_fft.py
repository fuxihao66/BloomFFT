# only support 1024 2048
import math
import numpy as np
import os
from tqdm import tqdm

os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import cv2
def PadToPowOfTwo(size):
    return pow(2, math.ceil(math.log2(size)))

COMPUTE_TYPE = np.float64
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
    return np.array([input0[0]*input1[0]-input0[1]*input1[1], input0[0]*input1[1]+input0[1]*input1[0]], dtype=COMPUTE_TYPE)
def Radix2FFT(input):
    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)

    i= 0
    Twiddle0 = np.array([math.cos(-2.*math.pi*i/(2.)), math.sin(-2.*math.pi*i/(2.))], dtype=COMPUTE_TYPE)
    Twiddle1 = np.array([math.cos(-2.*math.pi*(i+1)/(2.)), math.sin(-2.*math.pi*(i+1)/(2.))], dtype=COMPUTE_TYPE)
    result[i] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle0))
    result[i + 1] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle1))# TODO: USE BUTTERFLY
    return result

def Radix4FFT(input):
    input0 = input[::2] # odd
    input1 = input[1::2] # even
    result0 = Radix2FFT(input0)
    result1 = Radix2FFT(input1)
    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)

    for i in range(2):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(4.)), math.sin(-2.*math.pi*i/(4.))], dtype=COMPUTE_TYPE)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 2] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix8FFT(input):
    input0 = input[::2] # 
    input1 = input[1::2] # 
    result0 = Radix4FFT(input0)
    result1 = Radix4FFT(input1)
    
    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)
    for i in range(4):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(8.)), math.sin(-2.*math.pi*i/(8.))], dtype=COMPUTE_TYPE)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 4] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix16FFT(input):
    # input is [16,2]

    input0 = input[::2] # even
    input1 = input[1::2] # odd
    result0 = Radix8FFT(input0)# even
    result1 = Radix8FFT(input1)# odd

    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)
    for i in range(8):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(16.)), math.sin(-2.*math.pi*i/(16.))], dtype=COMPUTE_TYPE)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 8] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result

def Radix2FFT_inverse(input):
    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)

    i= 0
    Twiddle0 = np.array([math.cos(2.*math.pi*i/(2.)), math.sin(2.*math.pi*i/(2.))], dtype=COMPUTE_TYPE)
    Twiddle1 = np.array([math.cos(2.*math.pi*(i+1)/(2.)), math.sin(2.*math.pi*(i+1)/(2.))], dtype=COMPUTE_TYPE)
    result[i] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle0))
    result[i + 1] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle1))# TODO: USE BUTTERFLY
    return result

def Radix4FFT_inverse(input):
    input0 = input[::2] # odd
    input1 = input[1::2] # even
    result0 = Radix2FFT_inverse(input0)
    result1 = Radix2FFT_inverse(input1)
    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)

    for i in range(2):
        Twiddle = np.array([math.cos(2.*math.pi*i/(4.)), math.sin(2.*math.pi*i/(4.))], dtype=COMPUTE_TYPE)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 2] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix8FFT_inverse(input):
    input0 = input[::2] # 
    input1 = input[1::2] # 
    result0 = Radix4FFT_inverse(input0)
    result1 = Radix4FFT_inverse(input1)
    
    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)
    for i in range(4):
        Twiddle = np.array([math.cos(2.*math.pi*i/(8.)), math.sin(2.*math.pi*i/(8.))], dtype=COMPUTE_TYPE)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 4] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix16FFT_inverse(input):
    # input is [16,2]

    input0 = input[::2] # even
    input1 = input[1::2] # odd
    result0 = Radix8FFT_inverse(input0)# even
    result1 = Radix8FFT_inverse(input1)# odd

    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)
    for i in range(8):
        Twiddle = np.array([math.cos(2.*math.pi*i/(16.)), math.sin(2.*math.pi*i/(16.))], dtype=COMPUTE_TYPE)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 8] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result


def DFT16(input):
    # input is [16,2]

    result = np.zeros(input.shape, dtype=COMPUTE_TYPE)
    for i in range(16):
        result[i, 0] = 0
        result[i, 1] = 0
        for j in range(16):
            temp = ComplexMul(input[j], np.array([math.cos(-2.*math.pi*i*j/(16.)), math.sin(-2.*math.pi*i*j/(16.))], dtype=COMPUTE_TYPE))
            result[i, 0] += temp[0]
            result[i, 1] += temp[1]
    return result

def FFT(paddedwidth, img_row):# w, c

    img_width = img_row.shape[0]
    output_frequency_row = np.zeros((paddedwidth+2, 4), dtype=COMPUTE_TYPE)

    sub_array_num = paddedwidth // 16 // 16

    mergeradix16_rg_inputs = []
    mergeradix16_ba_inputs = []

    sub_sub_array_num = paddedwidth // 16
    for i in range(sub_sub_array_num):
        input = np.zeros((16, 2), dtype=COMPUTE_TYPE)
        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=COMPUTE_TYPE)
            else:
                input[j] = img_row[pixelIndex, 0:2]
        mergeradix16_rg_inputs.append(Radix16FFT(input))

        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=COMPUTE_TYPE)
                # if horizontal:
                #     input[:,1] = 1.# set alpha to 1
            else:
                input[j] = img_row[pixelIndex, 2:4]
        mergeradix16_ba_inputs.append(Radix16FFT(input))

    T_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
    T_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)

    F_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
    F_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)

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
        X_in_rg_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
        X_in_rg_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)
        X_in_ba_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
        X_in_ba_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)

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
        X_in_rg_real = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        X_in_rg_imag = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        X_in_ba_real = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        X_in_ba_imag = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)

        T_real = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        T_imag = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)

        F_real = np.zeros((sub_array_num, sub_array_num), dtype=COMPUTE_TYPE)
        F_imag = np.zeros((sub_array_num, sub_array_num), dtype=COMPUTE_TYPE)

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
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth][0:2] = np.array([ #g[0] 
        #             rg_imag[0], 0.
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth + 1][0:2] = np.array([ #g[N/2]
        #             -0.5*rg_imag[paddedwidth // 2]-0.5*rg_imag[paddedwidth // 2], 0.5*rg_real[paddedwidth // 2]-0.5*rg_real[paddedwidth // 2]# 0#TODO:
        #             ], dtype=COMPUTE_TYPE)
        #     else:
        #         output_frequency_row[i][0:2] = np.array([
        #             0.5*rg_real[i]+0.5*rg_real[paddedwidth-i], 0.5*rg_imag[paddedwidth-i] - 0.5*rg_imag[i],
        #             ], dtype=COMPUTE_TYPE)
                
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][0:2] = np.array([
        #             0.5*rg_imag[i + paddedwidth // 2 + 1]+0.5*rg_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*rg_real[i + paddedwidth // 2 + 1]-0.5*rg_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=COMPUTE_TYPE)# TODO:
        # ba
        ba_real = (Aa_ba - Bb_ba).flatten()
        ba_imag = (Ba_ba + Ab_ba).flatten()
        # for i in range(paddedwidth // 2 + 1):
        #     if i == 0:
        #         output_frequency_row[i][2:4] = np.array([ # r[0]
        #             ba_real[0], 0.
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth][2:4] = np.array([ #g[0]
        #             ba_imag[0], 0.
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth + 1][2:4] = np.array([ #g[N/2]
        #             -0.5*ba_imag[paddedwidth // 2]-0.5*ba_imag[paddedwidth // 2], 0.5*ba_real[paddedwidth // 2]-0.5*ba_real[paddedwidth // 2]# 0
        #             ], dtype=COMPUTE_TYPE)

        #     else:
        #         output_frequency_row[i][2:4] = np.array([
        #             0.5*ba_real[i]+0.5*ba_real[paddedwidth-i], 0.5*ba_imag[paddedwidth-i]-0.5*ba_imag[i],
        #             ], dtype=COMPUTE_TYPE)
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][2:4] = np.array([
        #             0.5*ba_imag[i + paddedwidth // 2 + 1]+0.5*ba_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*ba_real[i + paddedwidth // 2 + 1]-0.5*ba_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=COMPUTE_TYPE)

        # return output_frequency_row

        return rg_real, rg_imag, ba_real, ba_imag
    # else:
    #     assert(False)
    #     return []

def FFT_inverse(paddedwidth, img_row):# w, c

    img_width = img_row.shape[0]
    output_frequency_row = np.zeros((paddedwidth+2, 4), dtype=COMPUTE_TYPE)

    sub_array_num = paddedwidth // 16 // 16

    mergeradix16_rg_inputs = []
    mergeradix16_ba_inputs = []

    sub_sub_array_num = paddedwidth // 16
    for i in range(sub_sub_array_num):
        input = np.zeros((16, 2), dtype=COMPUTE_TYPE)
        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=COMPUTE_TYPE)
            else:
                input[j] = img_row[pixelIndex, 0:2]
        mergeradix16_rg_inputs.append(Radix16FFT_inverse(input))

        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=COMPUTE_TYPE)
                # if horizontal:
                #     input[:,1] = 1.# set alpha to 1
            else:
                input[j] = img_row[pixelIndex, 2:4]
        mergeradix16_ba_inputs.append(Radix16FFT_inverse(input))

    T_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
    T_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)

    F_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
    F_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)

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
        X_in_rg_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
        X_in_rg_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)
        X_in_ba_real = np.zeros((16, 16), dtype=COMPUTE_TYPE)
        X_in_ba_imag = np.zeros((16, 16), dtype=COMPUTE_TYPE)

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
        X_in_rg_real = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        X_in_rg_imag = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        X_in_ba_real = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        X_in_ba_imag = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)

        T_real = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)
        T_imag = np.zeros((sub_array_num, 256), dtype=COMPUTE_TYPE)

        F_real = np.zeros((sub_array_num, sub_array_num), dtype=COMPUTE_TYPE)
        F_imag = np.zeros((sub_array_num, sub_array_num), dtype=COMPUTE_TYPE)

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
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth][0:2] = np.array([ #g[0] 
        #             rg_imag[0], 0.
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth + 1][0:2] = np.array([ #g[N/2]
        #             -0.5*rg_imag[paddedwidth // 2]-0.5*rg_imag[paddedwidth // 2], 0.5*rg_real[paddedwidth // 2]-0.5*rg_real[paddedwidth // 2]# 0#TODO:
        #             ], dtype=COMPUTE_TYPE)
        #     else:
        #         output_frequency_row[i][0:2] = np.array([
        #             0.5*rg_real[i]+0.5*rg_real[paddedwidth-i], 0.5*rg_imag[paddedwidth-i] - 0.5*rg_imag[i],
        #             ], dtype=COMPUTE_TYPE)
                
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][0:2] = np.array([
        #             0.5*rg_imag[i + paddedwidth // 2 + 1]+0.5*rg_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*rg_real[i + paddedwidth // 2 + 1]-0.5*rg_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=COMPUTE_TYPE)# TODO:
        # ba
        ba_real = (Aa_ba - Bb_ba).flatten()
        ba_imag = (Ba_ba + Ab_ba).flatten()
        # for i in range(paddedwidth // 2 + 1):
        #     if i == 0:
        #         output_frequency_row[i][2:4] = np.array([ # r[0]
        #             ba_real[0], 0.
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth][2:4] = np.array([ #g[0]
        #             ba_imag[0], 0.
        #             ], dtype=COMPUTE_TYPE)
        #         output_frequency_row[paddedwidth + 1][2:4] = np.array([ #g[N/2]
        #             -0.5*ba_imag[paddedwidth // 2]-0.5*ba_imag[paddedwidth // 2], 0.5*ba_real[paddedwidth // 2]-0.5*ba_real[paddedwidth // 2]# 0
        #             ], dtype=COMPUTE_TYPE)

        #     else:
        #         output_frequency_row[i][2:4] = np.array([
        #             0.5*ba_real[i]+0.5*ba_real[paddedwidth-i], 0.5*ba_imag[paddedwidth-i]-0.5*ba_imag[i],
        #             ], dtype=COMPUTE_TYPE)
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][2:4] = np.array([
        #             0.5*ba_imag[i + paddedwidth // 2 + 1]+0.5*ba_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*ba_real[i + paddedwidth // 2 + 1]-0.5*ba_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ], dtype=COMPUTE_TYPE)

        # return output_frequency_row

        return rg_real, rg_imag, ba_real, ba_imag
def HorizontalFFTAndSplitOneForTwo(img):#
    h, w, _ = img.shape
    paddedWidth = PadToPowOfTwo(w)

    output_img = np.zeros((h, paddedWidth+2, 4), dtype=COMPUTE_TYPE)
    
    for col_index in tqdm(range(h)):
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedWidth, img[col_index,:])


        for i in range(paddedWidth // 2 + 1):# r0 to r[N/2], g[N/2+1]... g[N-1] g[0] g[N/2]
            if i == 0:
                output_img[col_index][i][0:2] = np.array([ # r[0]
                    rg_real[0], 0.
                    ], dtype=COMPUTE_TYPE)
                output_img[col_index][paddedWidth][0:2] = np.array([ #g[0] 
                    rg_imag[0], 0.
                    ], dtype=COMPUTE_TYPE)
                output_img[col_index][paddedWidth + 1][0:2] = np.array([ #g[N/2]
                    rg_imag[paddedWidth // 2], 0
                    ], dtype=COMPUTE_TYPE)
            else:
                output_img[col_index][i][0:2] = np.array([
                    0.5*rg_real[i]+0.5*rg_real[paddedWidth-i], 0.5*rg_imag[i] - 0.5*rg_imag[paddedWidth-i],
                    ], dtype=COMPUTE_TYPE)
                
        for i in range(paddedWidth // 2 - 1): #g[N/2+1]... g[N-1]
            output_img[col_index][i + paddedWidth // 2 + 1][0:2] = np.array([
                    0.5*rg_imag[i + paddedWidth // 2 + 1]+0.5*rg_imag[paddedWidth-(i + paddedWidth // 2 + 1)], -0.5*rg_real[i + paddedWidth // 2 + 1]+0.5*rg_real[paddedWidth-(i + paddedWidth // 2 + 1)], 
                    ], dtype=COMPUTE_TYPE)
             
        for i in range(paddedWidth // 2 + 1):
            if i == 0:
                output_img[col_index][i][2:4] = np.array([ # r[0]
                    ba_real[0], 0.
                    ], dtype=COMPUTE_TYPE)
                output_img[col_index][paddedWidth][2:4] = np.array([ #g[0]
                    ba_imag[0], 0.
                    ], dtype=COMPUTE_TYPE)
                output_img[col_index][paddedWidth + 1][2:4] = np.array([ #g[N/2]
                    ba_imag[paddedWidth // 2], 0
                    ], dtype=COMPUTE_TYPE)

            else:
                output_img[col_index][i][2:4] = np.array([
                    0.5*ba_real[i]+0.5*ba_real[paddedWidth-i], 0.5*ba_imag[i] - 0.5*ba_imag[paddedWidth-i],
                    ], dtype=COMPUTE_TYPE)
        for i in range(paddedWidth // 2 - 1): #g[N/2+1]... g[N-1]
            output_img[col_index][i + paddedWidth // 2 + 1][2:4] = np.array([
                    0.5*ba_imag[i + paddedWidth // 2 + 1]+0.5*ba_imag[paddedWidth-(i + paddedWidth // 2 + 1)], -0.5*ba_real[i + paddedWidth // 2 + 1]+0.5*ba_real[paddedWidth-(i + paddedWidth // 2 + 1)], 
                    ], dtype=COMPUTE_TYPE)
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

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=COMPUTE_TYPE)
    rgb_channel4[:,:,:3] = rgb_img

    h, w, _ = rgb_channel4.shape

    paddedWidth = PadToPowOfTwo(w)

    output_img = np.zeros((h, paddedWidth+2, 4), dtype=COMPUTE_TYPE)
    for i in tqdm(range(h)):
        paddedInput = np.zeros((paddedWidth, 4), dtype=COMPUTE_TYPE)
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

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=COMPUTE_TYPE)
    rgb_channel4[:,:,:3] = rgb_img

    h, w, _ = rgb_channel4.shape

    paddedWidth = PadToPowOfTwo(w)
    paddedHeight = PadToPowOfTwo(h)

    output_img = np.zeros((h, paddedWidth, 4), dtype=COMPUTE_TYPE)
    
    for i in tqdm(range(h)):
        paddedInput = np.zeros((paddedWidth, 4), dtype=COMPUTE_TYPE)
        paddedInput[:w] = rgb_channel4[i,:]
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedWidth, rgb_channel4[i,:])

        output_img[i,:,0] = rg_real
        output_img[i,:,1] = rg_imag
        output_img[i,:,2] = ba_real
        output_img[i,:,3] = ba_imag


    output_fft_img = np.zeros((paddedHeight, paddedWidth, 4), dtype=COMPUTE_TYPE)
    
    for j in tqdm(range(paddedWidth)):
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedHeight, output_img[:,j])

        output_fft_img[:,j,0] = rg_real
        output_fft_img[:,j,1] = rg_imag
        output_fft_img[:,j,2] = ba_real
        output_fft_img[:,j,3] = ba_imag

    paddedInput2D = np.zeros((paddedHeight, paddedWidth, 4), dtype=COMPUTE_TYPE)
    paddedInput2D[:h,:w,:] = rgb_channel4
    ref_fft_rg = np.fft.fft2(paddedInput2D[:,:,0] + 1j * paddedInput2D[:,:,1])
    ref_fft_ba = np.fft.fft2(paddedInput2D[:,:,2] + 1j * paddedInput2D[:,:,3])
    output_fft_ref = np.zeros(output_fft_img.shape, dtype=COMPUTE_TYPE)
    output_fft_ref[:,:,0] = np.real(ref_fft_rg)
    output_fft_ref[:,:,1] = np.imag(ref_fft_rg)
    output_fft_ref[:,:,2] = np.real(ref_fft_ba)
    output_fft_ref[:,:,3] = np.imag(ref_fft_ba)

    relative_error = np.zeros(output_fft_img.shape, dtype=COMPUTE_TYPE)
    relative_error[output_fft_ref == 0.] = np.abs(output_fft_img[output_fft_ref == 0.])
    relative_error[output_fft_ref != 0.] = np.abs(output_fft_img[output_fft_ref != 0.] - output_fft_ref[output_fft_ref != 0.]) / np.abs(output_fft_ref[output_fft_ref != 0.])
    

    print(np.average(relative_error))

def ConvertToLuma(ColorValue):
    return np.dot(ColorValue, np.array([0.2126, 0.7152, 0.0722], dtype=COMPUTE_TYPE))
def FilterPixel(Filter, PixelValue):
    bIsChanged = False
    Luma = ConvertToLuma(PixelValue[0:3])
    NewPixelValue = PixelValue.copy()
    if (Luma > Filter[0]):
        TargetLuma = Filter[2] * (Luma - Filter[0]) + Filter[0]
        TargetLuma  = min(TargetLuma, Filter[1])
        NewPixelValue[0:3] *= (TargetLuma / Luma)
        bIsChanged = True
    return NewPixelValue, bIsChanged
def HorizontalFFTTwoForOne_UnitTest():
    BrightPixelGain = np.array([7., 15000., 15.], dtype=COMPUTE_TYPE)
    input_img = cv2.imread("img/input_1280x720.exr", cv2.IMREAD_UNCHANGED)
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=COMPUTE_TYPE)
    rgb_channel4[:,:,:3] = rgb_img

    for i in range(rgb_channel4.shape[0]):
        for j in range(rgb_channel4.shape[1]):
            r, changed = FilterPixel(BrightPixelGain, rgb_channel4[i,j])
            if changed:
                rgb_channel4[i,j] = r

    horizontal_output = HorizontalFFTAndSplitOneForTwo(rgb_channel4)

    converted = horizontal_output.copy()
    converted[:,:,0] = horizontal_output[:,:,2]
    converted[:,:,2] = horizontal_output[:,:,0]
    # cv2.imwrite("horizontal_output.exr", converted.astype(np.float32))

    ref_img = cv2.imread("img/horizontal_output.exr", cv2.IMREAD_UNCHANGED)

    error = np.abs(converted.astype(np.float32) - ref_img)
    # error[ref_img > 1e-6] = error[ref_img > 1e-6] / np.abs(ref_img[ref_img > 1e-6])

    cv2.imwrite("error.exr", error)

# Y(k) = X1(K) + iX2(K)
# X1(N/2-1) = congegate(X1(N/2+1))
# X2(N/2-1) = congegate(X2(N/2+1))
def LoadFromTwoForOne(paddedWidth, input_array):
    output_array = np.zeros((paddedWidth, 4), dtype=COMPUTE_TYPE) 

    for i in range(paddedWidth):
        if i == 0:
            output_array[0,0] = input_array[0, 0]
            output_array[0,1] = input_array[paddedWidth, 0]
            output_array[0,2] = input_array[0, 2]
            output_array[0,3] = input_array[paddedWidth, 2]
        elif i == paddedWidth // 2:
            output_array[i,0] = input_array[i, 0]
            output_array[i,1] = input_array[paddedWidth+1, 0]
            output_array[i,2] = input_array[i, 2]
            output_array[i,3] = input_array[paddedWidth+1, 2]
        elif i < paddedWidth// 2:
            output_array[i,0] = input_array[i, 0] + input_array[paddedWidth - i, 1]
            output_array[i,1] = input_array[i, 1] + input_array[paddedWidth - i, 0]
            output_array[i,2] = input_array[i, 2] + input_array[paddedWidth - i, 3]
            output_array[i,3] = input_array[i, 3] + input_array[paddedWidth - i, 2]
        else:
            output_array[i,0] = -input_array[i, 1] + input_array[paddedWidth - i, 0]
            output_array[i,1] = input_array[i, 0] - input_array[paddedWidth - i, 1]
            output_array[i,2] = -input_array[i, 3] + input_array[paddedWidth - i, 2]
            output_array[i,3] = input_array[i, 2] - input_array[paddedWidth - i, 3]
    return output_array
def HorizontalFFTTwoForOne_Vertical_MulFilter_InvVertical_UnitTest():
    BrightPixelGain = np.array([7., 15000., 15.], dtype=COMPUTE_TYPE)
    input_img = cv2.imread("img/input_1280x720.exr", cv2.IMREAD_UNCHANGED)
    kernel_imag = cv2.imread("img/kernel.exr", cv2.IMREAD_UNCHANGED)
    
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=COMPUTE_TYPE)
    rgb_channel4[:,:,:3] = rgb_img

    for i in range(rgb_channel4.shape[0]):
        for j in range(rgb_channel4.shape[1]):
            r, changed = FilterPixel(BrightPixelGain, rgb_channel4[i,j])
            if changed:
                rgb_channel4[i,j] = r

    horizontal_output = HorizontalFFTAndSplitOneForTwo(rgb_channel4)

    # flip r b channel
    converted = kernel_imag.copy()
    converted[:,:,0] = kernel_imag[:,:,2]
    converted[:,:,2] = kernel_imag[:,:,0]
    kernel_imag = converted

    paddedLength = max(PadToPowOfTwo(horizontal_output.shape[0]), kernel_imag.shape[0]) 

    filtered_horizontal_output = horizontal_output.copy()

    # [r, b] [g, a]
    kernelsum = [[kernel_imag[0, 0][0], kernel_imag[0, 0][2]], [kernel_imag[0, kernel_imag.shape[1]-2][0], kernel_imag[0, kernel_imag.shape[1]-2][2]]]  # x and z channel
    
    for col_index in tqdm(range(horizontal_output.shape[1])):
        rg_real, rg_imag, ba_real, ba_imag = FFT(paddedLength, horizontal_output[:,col_index,:])

        ComplexFFTResult_rg = rg_real + 1j * rg_imag
        ComplexKernel_rg = kernel_imag[:,col_index,0] + 1j * kernel_imag[:,col_index,1]

        ComplexFFTResult_ba = ba_real + 1j * ba_imag
        ComplexKernel_ba = kernel_imag[:,col_index,2] + 1j * kernel_imag[:,col_index,3]

        Filtered_rg_complex = ComplexFFTResult_rg * ComplexKernel_rg
        Filtered_ba_complex = ComplexFFTResult_ba * ComplexKernel_ba

        weights = [kernelsum[0][0], kernelsum[0][1]]# TODO: different from ue5, check
        if 2 * col_index > horizontal_output.shape[1] - 2:
            weights = [kernelsum[1][0], kernelsum[1][1]]

        Filtered = np.zeros((paddedLength, 4), dtype=COMPUTE_TYPE)
        Filtered[:,0] = np.real(Filtered_rg_complex) / weights[0] # r/g
        Filtered[:,1] = np.imag(Filtered_rg_complex) / weights[0] # r/g
        Filtered[:,2] = np.real(Filtered_ba_complex) / weights[1] # b/a
        Filtered[:,3] = np.imag(Filtered_ba_complex) / weights[1] # b/a
        rg_real_inv, rg_imag_inv, ba_real_inv, ba_imag_inv = FFT_inverse(paddedLength, Filtered)

        filtered_horizontal_output[:,col_index,0] = rg_real_inv[:filtered_horizontal_output.shape[0]] * (1. / paddedLength)
        filtered_horizontal_output[:,col_index,1] = rg_imag_inv[:filtered_horizontal_output.shape[0]] * (1. / paddedLength)
        filtered_horizontal_output[:,col_index,2] = ba_real_inv[:filtered_horizontal_output.shape[0]] * (1. / paddedLength)
        filtered_horizontal_output[:,col_index,3] = ba_imag_inv[:filtered_horizontal_output.shape[0]] * (1. / paddedLength)
    return filtered_horizontal_output
def InvHorizontal_UnitTest():
    DstPostFilterParameter = np.array([0.01738,  0.0174,  0.01864,  0.00602], dtype=COMPUTE_TYPE)
    input_img = cv2.imread("img/input_1280x720.exr", cv2.IMREAD_UNCHANGED)

    filtered_horizontal_output = cv2.imread("filtered_horizontal.exr", cv2.IMREAD_UNCHANGED)# saved before channel fliped 
    # filtered_horizontal_output = cv2.imread("img/filtered_horizontal_output.exr", cv2.IMREAD_UNCHANGED)
    # converted_filtered = filtered_horizontal_output.copy()
    # converted_filtered[:,:,0] = filtered_horizontal_output[:,:,2]
    # converted_filtered[:,:,2] = filtered_horizontal_output[:,:,0]
    # filtered_horizontal_output = converted_filtered
    # split
    paddedWidth = PadToPowOfTwo(input_img.shape[1])
    result_img = np.zeros(input_img.shape, dtype=COMPUTE_TYPE)
    for row_index in tqdm(range(input_img.shape[0])):
        input_array = LoadFromTwoForOne(paddedWidth, filtered_horizontal_output[row_index,:]) 
        
        rg_real, rg_imag, ba_real, ba_imag = FFT_inverse(paddedWidth, input_array)
        
        result_img[row_index,:,0] = rg_real[:input_img.shape[1]] * (1. / paddedWidth) * DstPostFilterParameter[0]
        result_img[row_index,:,1] = rg_imag[:input_img.shape[1]] * (1. / paddedWidth) * DstPostFilterParameter[1]
        result_img[row_index,:,2] = ba_real[:input_img.shape[1]] * (1. / paddedWidth) * DstPostFilterParameter[0] # TODO: copy from ue5, might be wrong
        result_img[row_index,:,3] = ba_imag[:input_img.shape[1]] * (1. / paddedWidth) * DstPostFilterParameter[3]
    converted_result = result_img.copy()
    converted_result[:,:,0] = result_img[:,:,2]
    converted_result[:,:,2] = result_img[:,:,0]
    converted_result[:,:,3] = 1.

    cv2.imwrite("final.exr", converted_result.astype(np.float32))

def HorizontalInvHorizontal_UnitTest():
    input_img = cv2.imread("img/input_1280x720.exr", cv2.IMREAD_UNCHANGED)
    
    rgb_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    rgb_channel4 = np.ones((rgb_img.shape[0], rgb_img.shape[1], 4), dtype=COMPUTE_TYPE)
    rgb_channel4[:,:,:3] = rgb_img

    horizontal_output = HorizontalFFTAndSplitOneForTwo(rgb_channel4)
    # split
    paddedWidth = PadToPowOfTwo(input_img.shape[1])
    result_img = np.zeros(input_img.shape, dtype=COMPUTE_TYPE)
    for row_index in tqdm(range(input_img.shape[0])):
        input_array = LoadFromTwoForOne(paddedWidth, horizontal_output[row_index,:]) 
        
        rg_real, rg_imag, ba_real, ba_imag = FFT_inverse(paddedWidth, input_array)
        
        result_img[row_index,:,0] = rg_real[:input_img.shape[1]] * (1. / paddedWidth)
        result_img[row_index,:,1] = rg_imag[:input_img.shape[1]] * (1. / paddedWidth)
        result_img[row_index,:,2] = ba_real[:input_img.shape[1]] * (1. / paddedWidth)
        result_img[row_index,:,3] = ba_imag[:input_img.shape[1]] * (1. / paddedWidth)
    converted_result = result_img.copy()
    converted_result[:,:,0] = result_img[:,:,2]
    converted_result[:,:,2] = result_img[:,:,0]
    converted_result[:,:,3] = 1.

    cv2.imwrite("horizontal_invhorizontal.exr", converted_result.astype(np.float32))


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
    # fft_result = np.zeros((paddedLength, 4), dtype=COMPUTE_TYPE)
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
    # HorizontalFFTTwoForOne_UnitTest()
    # test3: vertical fft 
    # TwoDimenFFTUnitTest()
    # test4: vertical fft and multiply

    # test5: inverse vertical fft
    
    # test6: inverse horizontal fft

    # HorizontalFFTTwoForOne_Vertical_MulFilter_InvVertical_UnitTest()
    # InvHorizontal_UnitTest()
    HorizontalInvHorizontal_UnitTest()