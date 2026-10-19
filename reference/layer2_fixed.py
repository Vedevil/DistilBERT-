"""Step 9 independent layer2 spec, retaining Q13 with 18-bit FFN lin2 weights.
All other tensor formats and activation arithmetic remain Step7. Explicit
sat32 at linear outputs; bound3072*2^31*131072+2^44 <2^63 proves int64 safe.
"""
import numpy as np
from reference.ffn_fixed import linear_wide,gelu_q11,round_away
from reference.transformer_wide_fixed import attention_out_wide
from reference.residual_norm_wide_fixed import residual_norm_wide_fixed


def linear_w18(x,w,b):
    x,w,b=map(np.asarray,(x,w,b))
    if x.dtype!=np.int32 or w.dtype!=np.int32 or b.dtype!=np.int32:raise ValueError('int32 storage required; weights bounded to signed18 bits')
    if x.ndim!=2 or w.ndim!=2 or b.shape!=(len(w),) or x.shape[1]!=w.shape[1] or not 1<=w.shape[1]<=3072:raise ValueError('bad shapes')
    if np.any(w< -131072) or np.any(w>131071):raise ValueError('weight outside int18')
    acc=x.astype(np.int64)@w.astype(np.int64).T+(b.astype(np.int64)<<13)
    raw=round_away(acc,13);out=np.clip(raw,-2147483648,2147483647).astype(np.int32)
    return out,int(np.count_nonzero(raw!=out))


def ffn_w18(x,w1,b1,w2,b2):
    z,c1=linear_wide(x,w1,b1);h=gelu_q11(z);y,c2=linear_w18(h,w2,b2)
    return z,h,y,{'lin1_saturated':c1,'lin2_saturated':c2}


def transformer_layer2_fixed(x,w,b,wo,bo,keep,g1,beta1,w1,b1,w2,b2,g2,beta2):
    r=attention_out_wide(x,w,b,wo,bo,keep)
    first=residual_norm_wide_fixed(x,r['attention'],g1,beta1)
    z,h,ff,fa=ffn_w18(first['output'],w1,b1,w2,b2)
    second=residual_norm_wide_fixed(first['output'],ff,g2,beta2)
    r.update(first=first,z=z,gelu=h,ffn=ff,second=second,output=second['output'])
    r['audit'].update(first_norm_saturated=first['saturated'],second_norm_saturated=second['saturated'],**fa)
    return r
