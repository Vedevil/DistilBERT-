"""Independent signed-integer specification for Step 4 attention residual + LayerNorm.

No NumPy floating-point operations in the bit-exact path. It normalizes the
Q4.11 int17 residual x_q11 + attention_out_q11, with variance in Q22,
minimum epsilon 1 Q22, rounded integer sqrt to a Q11 standard deviation,
and rounded unsigned reciprocal in Q20. Gamma is signed Q2.13 int16,
beta is signed Q4.11 int32, output signed Q20.11 int32 (NOT int16:
actual checkpoint channel 308 exceeds the Q4.11 int16 range).

This differs from PyTorch's float LayerNorm eps=1e-12 near zero variance:
one Q22 is ~2.38e-7, deliberately preventing a zero reciprocal denominator.
"""
from __future__ import annotations

import math

import numpy as np

EPS_Q22=1


def saturate_i32(value: int) -> int:
    return max(-(1<<31),min((1<<31)-1,int(value)))


def round_div_away(numerator: int, denominator: int) -> int:
    """Signed nearest integer division; ties away from zero."""
    if denominator<=0:raise ValueError('positive denominator required')
    numerator=int(numerator)
    mag=(abs(numerator)+denominator//2)//denominator
    return -mag if numerator<0 else mag


def sqrt_nearest_u32(value: int) -> int:
    if not 0<=value<=0xffffffff:raise ValueError('uint32 radicand required')
    floor=math.isqrt(value)
    remainder=value-floor*floor
    return min(65535,floor+int(remainder>floor))


def residual_norm_fixed(x_q11,attention_q11,gamma_q13,beta_q11):
    """Return (int17 residual, mean Q11, variance Q22, std Q11,
    unsigned inverse std Q20, signed int32 normalized LayerNorm) per token.
    Shape x/y [N,D], gamma/beta [D]; weights must be appropriately quantized.
    """
    x,y=np.asarray(x_q11),np.asarray(attention_q11)
    gamma,beta=np.asarray(gamma_q13),np.asarray(beta_q11)
    if (x.dtype!=np.int16 or y.dtype!=np.int16 or gamma.dtype!=np.int16 or
        beta.dtype!=np.int32 or x.ndim!=2 or x.shape!=y.shape or
        gamma.shape!=(x.shape[1],) or beta.shape!=gamma.shape or
        not 1<=x.shape[1]<=3072):
        raise ValueError('expected Q11 int16 x/y [N,D], Q13 int16 gamma, int32 Q11 beta; 1<=D<=3072')
    n,dim=x.shape
    residual=x.astype(np.int32)+y.astype(np.int32)
    output=np.empty((n,dim),dtype=np.int32)
    means=np.empty(n,dtype=np.int32)
    variances=np.empty(n,dtype=np.uint32)
    stds=np.empty(n,dtype=np.uint16)
    inverses=np.empty(n,dtype=np.uint32)
    for t in range(n):
        values=[int(v) for v in residual[t]]
        mean=round_div_away(sum(values),dim)
        square_sum=sum((value-mean)**2 for value in values)
        variance=(square_sum+dim//2)//dim+EPS_Q22
        variance=min(variance,0xffffffff)
        std=max(1,sqrt_nearest_u32(variance))
        inverse=(1<<31)+std//2
        inverse//=std
        means[t]=mean
        variances[t]=variance
        stds[t]=std
        inverses[t]=inverse
        for f,value in enumerate(values):
            norm_q15=round_div_away((value-mean)*inverse,1<<16)
            scaled_q11=round_div_away(norm_q15*int(gamma[f]),1<<17)
            output[t,f]=saturate_i32(scaled_q11+int(beta[f]))
    return residual,means,variances,stds,inverses,output
