#https://github.com/clevercool/ANT_Micro22/blob/2cc589c4b5b2a936737d6afba0eb27dbd41eb88d/ant_quantization/antquant/quant_modules.py

#https://intellabs.github.io/distiller/algo_quantization.html

import torch
import numpy as np

#alg 2 from ANT paper
#take tensor and return best datatype based on distributio
def best_datatype(t):
    candidates=['int4', 'float4', 'PoT4', 'flint4']




#alg 1 from ANT paper
#take 32 bit float tensor and quantization datatype and return tensor quantized then dequantized by to 32 bit float
#see page 1416 equation 2
def quantize_tensor(t, datatype):
    #TODO how to get the right scale factors?
    
    if datatype == 'int4':
        #[-8, 7]
        min_range = -8
        max_range = 7
        scale_factor = torch.max(torch.abs(t)) * 2 / 16  

        quant_fn = torch.round
        dequant_fn = lambda x: x
    elif datatype == 'float4':
        #TODO how are bits divided up among 4 bit float?
        #TODO section V.B page 1421, float is not used?
        pass
    elif datatype == 'PoT4':
        #TODO what is the bias? using 0 for now
        #[-2^14, 2^14]
        min_range = -2**14
        max_range = 2**14
        scale_factor = torch.max(torch.abs(t)) / (2**14)
        # scale_factor = torch.max(torch.abs(t)) * 2 / 16 

        def quant_fn(x):
            out = torch.pow(2, torch.round(torch.log2(torch.abs(x))))
            out = out * torch.sign(x)
            return out
        
        dequant_fn = lambda x: x

    elif datatype == 'flint4':
        #signed version page 1422
        # TODO bias -1?
        # TODO double use of 0 and -0 for values?

        # first bit is sign
        # flint         base10
        # 000   -       0
        # 001   2^0     1
        # 01x   2^1     2,3
        # 11x   2^2     4,6
        # 101   2^3     8
        # 100   2^4     16

        min_range = -16
        max_range = 16
        scale_factor = torch.max(torch.abs(t)) / 16

        def quant_fn(x):
            #TODO this is a bit hacked only works for 4 bit signed flint
            unsigned = torch.abs(torch.round(x))
            exp_index = (torch.floor(torch.log2(unsigned)) + 1).to(torch.uint8)

            tmp_len = torch.logical_or(exp_index == 2, exp_index == 3)
            exp_len = (2 * tmp_len) + (3 * (torch.logical_not(tmp_len)))

            #sign bit, 0=+, 1=-
            sign = (x < 0)

            #exp bits
            exp = torch.bitwise_left_shift(0b1, exp_len - 1) * (exp_index >=3)
            exp = exp + (exp_index < 5)

            #mantissa bit
            mantissa_len = 3 - exp_len
            mantissa = mantissa_round((unsigned / torch.pow(2, exp_index-1) - 1) * 2**mantissa_len).to(torch.uint8)

            #concatenate sign, exponent, mantissa
            out = concat(sign, exp, exp_len)
            out = concat(out, mantissa, mantissa_len)

            #return 0000 on 0 input
            out = out * (unsigned != 0)

            return out

        def dequant_fn(x):
            #TODO this is also a bit hacked only works for signed 4 bit flint
            sign = torch.bitwise_and(torch.bitwise_right_shift(x, 3), 0b1)
            sign = -1 * (sign) + 1 * (1 - sign) #convert 1,0 to -1, 1
            sign = sign * (x != 0b0) #return 0 on 0b0000

            unsigned = torch.bitwise_and(x, 0b111)
            bit_2 = torch.bitwise_right_shift(unsigned, 2)

            #equation 3 of paper
            leading_zeros = lzd(torch.bitwise_and(unsigned, 0b11))
    
            exp0 = 1 - leading_zeros
            exp1 = 2 + leading_zeros

            exp = (bit_2 == 0)*exp0 + (bit_2 == 1)*exp1

            #get mantissa value
            mantissa = 1 + (leading_zeros == 0) * (torch.bitwise_and(unsigned, 0b1)) * 0.5

            #combine exp and mantissa
            return sign * torch.pow(2, exp) * mantissa


    # scale min value to 0 and max value to max value that can be represented
    # TODO determine scale factor with range clipping

    # equation 2
    t_out = t / scale_factor 
    # print(t_out)
    # print(t)
    t_out = quant_fn(t_out)
    # print([np.binary_repr(x) for x in t_out])
    t_out = torch.clamp(t_out, min_range, max_range)
    t_out = dequant_fn(t_out)
    # print(t_out)
    t_out = t_out * scale_factor

    return t_out


#scale factor 1424 determined by STE mothod?



def concat(x, y, y_len):
    return torch.bitwise_or(torch.bitwise_left_shift(x, y_len), y)

def mantissa_round(x):
    #round x.5 down to x instead of even
    return torch.ceil(x - 0.5)

def lzd(x):
    #takes two bit input
    clean = torch.bitwise_and(x.to(torch.uint8), 0b11)
    out = torch.zeros(clean.size())

    out = out + (torch.bitwise_right_shift(clean, 1) == 0)
    out = out + (clean == 0b00)
    return out


if __name__ == '__main__':
    t1 = torch.arange(-16,17)

    # print(t1)
    # print(quantize_tensor(t1, 'int4'))
    # print()
    # print(t1)
    # print(quantize_tensor(t1, 'PoT4'))
    # print()
    # print(t1)
    # print(quantize_tensor(t1, 'flint4'))
    # tmp()
    quantize_tensor(t1, 'flint4')

    # for i in range(0,4):
    #     print(i)
    #     print(lzd(torch.Tensor([i])))
