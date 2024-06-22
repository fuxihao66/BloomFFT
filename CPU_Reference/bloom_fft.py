# only support 1024 2048
import cv2
import math
import numpy as np
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
    return np.array([input0[0]*input1[0]-input0[1]*input1[1], input0[0]*input1[1]+input0[1]*input1[0]], dtype=np.float32)
def Radix2FFT(input):
    result = np.zeros(input.shape, dtype=np.float32)

    i= 0
    Twiddle0 = np.array([math.cos(-2.*math.pi*i/(2.)), math.sin(-2.*math.pi*i/(2.))], dtype=np.float32)
    Twiddle1 = np.array([math.cos(-2.*math.pi*(i+1)/(2.)), math.sin(-2.*math.pi*(i+1)/(2.))], dtype=np.float32)
    result[i] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle0))
    result[i + 1] = ComplexAdd(input[0], ComplexMul(input[1], Twiddle1))# TODO: USE BUTTERFLY
    return result

def Radix4FFT(input):
    input0 = input[::2] # odd
    input1 = input[1::2] # even
    result0 = Radix2FFT(input0)
    result1 = Radix2FFT(input1)
    result = np.zeros(input.shape, dtype=np.float32)

    for i in range(2):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(4.)), math.sin(-2.*math.pi*i/(4.))], dtype=np.float32)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 2] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix8FFT(input):
    input0 = input[::2] # 
    input1 = input[1::2] # 
    result0 = Radix4FFT(input0)
    result1 = Radix4FFT(input1)
    
    result = np.zeros(input.shape, dtype=np.float32)
    for i in range(4):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(8.)), math.sin(-2.*math.pi*i/(8.))], dtype=np.float32)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 4] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result
def Radix16FFT(input):
    # input is [16,2]

    input0 = input[::2] # even
    input1 = input[1::2] # odd
    result0 = Radix8FFT(input0)# even
    result1 = Radix8FFT(input1)# odd

    result = np.zeros(input.shape, dtype=np.float32)
    for i in range(8):
        Twiddle = np.array([math.cos(-2.*math.pi*i/(16.)), math.sin(-2.*math.pi*i/(16.))], dtype=np.float32)
        result[i] = ComplexAdd(result0[i], ComplexMul(result1[i], Twiddle))
        result[i + 8] = ComplexSub(result0[i], ComplexMul(result1[i], Twiddle))
    return result

def DFT16(input):
    # input is [16,2]

    result = np.zeros(input.shape, dtype=np.float32)
    for i in range(16):
        result[i, 0] = 0
        result[i, 1] = 0
        for j in range(16):
            temp = ComplexMul(input[j], np.array([math.cos(-2.*math.pi*i*j/(16.)), math.sin(-2.*math.pi*i*j/(16.))], dtype=np.float32))
            result[i, 0] += temp[0]
            result[i, 1] += temp[1]
    return result
def FFT(paddedwidth, img_row):# w, c

    img_width = img_row.shape[0]
    output_frequency_row = np.zeros((paddedwidth+2, 4), dtype=np.float32)

    sub_array_num = paddedwidth // 16 // 16

    mergeradix16_rg_inputs = []
    mergeradix16_ba_inputs = []

    sub_sub_array_num = paddedwidth // 16
    for i in range(sub_sub_array_num):
        input = np.zeros((16, 2), dtype=np.float32)
        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=np.float32)
            else:
                input[j] = img_row[pixelIndex, 0:2]
        mergeradix16_rg_inputs.append(Radix16FFT(input))

        for j in range(16):
            pixelIndex = sub_sub_array_num * j + i
            if pixelIndex >= img_width:
                input[j] = np.zeros((2), dtype=np.float32)
            else:
                input[j] = img_row[pixelIndex, 2:4]
        mergeradix16_ba_inputs.append(Radix16FFT(input))

    T_real = np.zeros((16, 16), dtype=np.float32)
    T_imag = np.zeros((16, 16), dtype=np.float32)

    F_real = np.zeros((16, 16), dtype=np.float32)
    F_imag = np.zeros((16, 16), dtype=np.float32)

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
        X_in_rg_real = np.zeros((16, 16), dtype=np.float32)
        X_in_rg_imag = np.zeros((16, 16), dtype=np.float32)
        X_in_ba_real = np.zeros((16, 16), dtype=np.float32)
        X_in_ba_imag = np.zeros((16, 16), dtype=np.float32)

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
    if sub_array_num == 4 or sub_array_num == 8:
        X_in_rg_real = np.zeros((sub_array_num, 256), dtype=np.float32)
        X_in_rg_imag = np.zeros((sub_array_num, 256), dtype=np.float32)
        X_in_ba_real = np.zeros((sub_array_num, 256), dtype=np.float32)
        X_in_ba_imag = np.zeros((sub_array_num, 256), dtype=np.float32)

        T_real = np.zeros((sub_array_num, 256), dtype=np.float32)
        T_imag = np.zeros((sub_array_num, 256), dtype=np.float32)

        F_real = np.zeros((sub_array_num, sub_array_num), dtype=np.float32)
        F_imag = np.zeros((sub_array_num, sub_array_num), dtype=np.float32)

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
        #             , 0.
        #             ])
        #         output_frequency_row[paddedwidth][0:2] = np.array([ #g[0]
        #             , 0.
        #             ])
        #         output_frequency_row[paddedwidth + 1][0:2] = np.array([ #g[N/2]
        #             -0.5*rg_imag[paddedwidth // 2]-0.5*rg_imag[paddedwidth // 2], 0.5*rg_real[paddedwidth // 2]-0.5*rg_real[paddedwidth // 2]
        #             ])
        #     else:
        #         output_frequency_row[i][0:2] = np.array([
        #             0.5*rg_real[i]+0.5*rg_real[paddedwidth-i], 0.5*rg_imag[i]-0.5*rg_imag[paddedwidth-i],
        #             ])
                
        # for i in range(paddedwidth // 2 - 1): #g[N/2+1]... g[N-1]
        #     output_frequency_row[i + paddedwidth // 2 + 1][0:2] = np.array([
        #             -0.5*rg_imag[i + paddedwidth // 2 + 1]-0.5*rg_imag[paddedwidth-(i + paddedwidth // 2 + 1)], 0.5*rg_real[i + paddedwidth // 2 + 1]-0.5*rg_real[paddedwidth-(i + paddedwidth // 2 + 1)], 
        #             ])# TODO:
        # ba
        ba_real = (Aa_ba - Bb_ba).flatten()
        ba_imag = (Ba_ba + Ab_ba).flatten()
        # for i in range(paddedwidth // 2 + 1):
        #     if i == 0:
        #         output_frequency_row[i][2:4] = 
        #         output_frequency_row[i + paddedwidth // 2 + 1][2:4] = 
        #     else:
        #         output_frequency_row[i][2:4] = np.array([
        #             0.5*ba_real[i]+0.5*ba_real[paddedwidth-i], 0.5*ba_imag[i]-0.5*ba_imag[paddedwidth-i],
        #             ])
        #         output_frequency_row[i + paddedwidth // 2 + 1][2:4] = np.array([
        #             -0.5*ba_imag[i]-0.5*ba_imag[paddedwidth-i], 0.5*ba_real[i]-0.5*ba_real[paddedwidth-i], 
        #             ])
        # return output_frequency_row

        return rg_real, rg_imag, ba_real, ba_imag
    else:
        assert(False)
        return []
    
def HorizontalFilter(img):
    h, w, _ = img.shape
    output_img = np.zeros((h, w+2, 4), dtype=np.float32)
    paddedWidth = PadToPowOfTwo(w)
    
    for i in range(h):
        frequency_row = FFT(paddedWidth, img[i,:])
        output_img[i,:] = frequency_row
    return output_img
def HorizontalFilter_UnitTest():
    input_img = cv2.imread("img/input_1280x720.hdr", cv2.IMREAD_UNCHANGED)

    horizontal_output = HorizontalFilter(input_img)
    cv2.imwrite("horizontal_output.exr", horizontal_output)
    
if __name__ == "__main__":
    random_array = np.random.random((256, 4))

    rg_real, rg_imag, ba_real, ba_imag = FFT(256, random_array)

    rg_ref = np.fft.fft(random_array[...,0] + 1j * random_array[...,1])
    ba_ref = np.fft.fft(random_array[...,2] + 1j * random_array[...,3])

    # print(rg_real)
    # print(rg_imag)
    # print(rg_ref)
    print(ba_real)
    print(ba_imag)
    print(ba_ref)

    # HorizontalFilter_UnitTest()

    # input = np.array([[0.5, 0.3], [0.3, 0.2], [0.5, 0.2], [0.45, 0.72], [0.35, 0.52], [0.15, 0.82], [0.25, 0.22], [0.15, 0.52], [0.25, 0.83], [0.23, 0.22], [0.85, 0.2], [0.35, 0.29], [0.52, 0.25], [0.25, 0.92], [0.5, 0.2], [0.59, 0.62]], dtype=np.float32)
    # # result = Radix16FFT(input)
    # # result = Radix2FFT(input)
    # # result = DFT16(input)
    # result = np.fft.fft(input[...,0] + 1j * input[...,1])
    # print(result)