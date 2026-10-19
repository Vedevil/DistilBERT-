"""Independent Step 5 integer specification; no HDL-derived expected outputs.

All activations, both linear outputs and GELU outputs are signed int32 Q11.
Weights int16 Q13; bias int32 Q11; accumulate safely in int64, nearest ties
away from zero then saturate int32. GELU approximates exact-erf GELU using a
normal-CDF Q20 table sampled at k/64 for k=0..512, linear interpolation and
identity/zero tails for |x|>=8. No dropout, residual or second LayerNorm.
"""
import math
import numpy as np

CDF_Q20=tuple(int(math.floor((0.5*(1+math.erf(k/64/math.sqrt(2))))*(1<<20)+0.5))
              for k in range(513))


def round_away(value,shift):
    v=np.asarray(value,dtype=np.int64)
    mag=(np.abs(v)+(1<<(shift-1)))>>shift
    return np.where(v<0,-mag,mag)


def quantize(array,frac,dtype):
    a=np.asarray(array,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('nonfinite tensor')
    raw=np.copysign(np.floor(np.abs(a)*(1<<frac)+0.5),a)
    lim=np.iinfo(dtype)
    return np.clip(raw,lim.min,lim.max).astype(dtype),int(np.count_nonzero((raw<lim.min)|(raw>lim.max)))


def linear_wide(x,w,b):
    x=np.asarray(x);w=np.asarray(w);b=np.asarray(b)
    if x.dtype!=np.int32 or w.dtype!=np.int16 or b.dtype!=np.int32:
        raise ValueError('int32 activation/bias and int16 weights required')
    if x.ndim!=2 or w.ndim!=2 or b.shape!=(w.shape[0],) or x.shape[1]!=w.shape[1] or not 1<=w.shape[1]<=3072:
        raise ValueError('invalid dimensions')
    acc=x.astype(np.int64)@w.astype(np.int64).T+(b.astype(np.int64)<<13)
    rounded=round_away(acc,13)
    return np.clip(rounded,-(1<<31),(1<<31)-1).astype(np.int32),int(np.count_nonzero((rounded<-(1<<31))|(rounded>(1<<31)-1)))


def gelu_q11(values):
    x=np.asarray(values)
    if x.dtype!=np.int32:raise ValueError('int32 required')
    v=x.astype(np.int64); a=np.abs(v)
    idx=np.minimum(a>>5,511);frac=a&31
    lut=np.asarray(CDF_Q20,dtype=np.int64)
    phi=lut[idx]+(((lut[idx+1]-lut[idx])*frac+16)>>5)
    phi=np.where(v<0,(1<<20)-phi,phi)
    y=round_away(v*phi,20)
    y=np.where(a>=16384,np.where(v<0,0,v),y)
    return y.astype(np.int32)


def ffn_fixed(x,w1,b1,w2,b2):
    z,c1=linear_wide(x,w1,b1)
    h=gelu_q11(z)
    y,c2=linear_wide(h,w2,b2)
    return z,h,y,{'lin1_saturated':c1,'lin2_saturated':c2}
