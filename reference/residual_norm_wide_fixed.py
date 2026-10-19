"""Step 6 independent integer spec for SECOND residual/LayerNorm.

Two signed int32 Q11 operands -> unsaturated signed int33 residual.
Rounded integer Q11 mean, centered population variance Q22 + epsilon=1,
rounded integer square root Q11, reciprocal Q44=round(2^55/std_q11).
Normalize to Q15, gamma int16 Q13, beta int32 Q11, output int32 Q11.
Python arbitrary integers in the statistic/product path avoid int64 squaring
or accumulation overflow. No variance saturation or narrowed input clipping.
Legal dimensions 1..3072. Float Torch eps=1e-12 is different near zero variance.
"""
import math
import numpy as np

EPS_Q22=1


def round_div(value,denominator):
    if denominator<=0:raise ValueError('positive denominator required')
    v=int(value);m=(abs(v)+denominator//2)//denominator
    return -m if v<0 else m


def residual_norm_wide_fixed(skip_q11,ffn_q11,gamma_q13,beta_q11):
    x,f,g,b=map(np.asarray,(skip_q11,ffn_q11,gamma_q13,beta_q11))
    if (x.dtype!=np.int32 or f.dtype!=np.int32 or g.dtype!=np.int16 or b.dtype!=np.int32
        or x.ndim!=2 or x.shape!=f.shape or not 1<=x.shape[1]<=3072
        or g.shape!=(x.shape[1],) or b.shape!=g.shape):
        raise ValueError('expected int32 [N,D] operands/beta, int16 gamma, 1<=D<=3072')
    n,d=x.shape
    residual=x.astype(np.int64)+f.astype(np.int64)
    out=np.empty((n,d),np.int32);means=np.empty(n,np.int64)
    variances=[];stds=[];inverses=[];norms=[]
    saturated=0
    for t in range(n):
        values=[int(v) for v in residual[t]]
        mean=round_div(sum(values),d)
        variance=round_div(sum((v-mean)**2 for v in values),d)+EPS_Q22
        root=math.isqrt(variance);std=max(1,root+int(variance-root*root>root))
        inv=round_div(1<<55,std)
        means[t]=mean;variances.append(variance);stds.append(std);inverses.append(inv)
        row=[]
        for i,v in enumerate(values):
            norm=round_div((v-mean)*inv,1<<40)
            assert -(1<<31)<=norm<(1<<31),'Q15 bound violated'
            row.append(norm)
            scaled=round_div(norm*int(g[i]),1<<17)+int(b[i])
            clamped=max(-(1<<31),min((1<<31)-1,scaled));saturated+=clamped!=scaled
            out[t,i]=clamped
        norms.append(row)
    return {'residual':residual,'mean':means,'variance':np.array(variances,dtype=object),
            'std':np.array(stds,dtype=np.int64),'inverse':np.array(inverses,dtype=np.int64),
            'norm_q15':np.array(norms,dtype=np.int32),'output':out,'saturated':int(saturated)}
