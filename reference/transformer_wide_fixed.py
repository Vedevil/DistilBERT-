"""Step 7 independent wide attention and reusable one-layer specification.
Q11 activations/int32, Q13 weights/int16, Q11 biases/int32, Q10 scores/int32.
Python arbitrary integers for QK/PV (never NumPy int64 QK matrix products).
All listed boundary saturation is explicit, counted, and tested. Not a global
model calibration, later-layer reference, or six-layer inference program.
"""
import numpy as np
from reference.fixed import round_shift_away,softmax_fixed
from reference.ffn_fixed import linear_wide,ffn_fixed
from reference.residual_norm_wide_fixed import residual_norm_wide_fixed


def sat32(v):
    return max(-(1<<31),min((1<<31)-1,int(v)))


def attention_wide(q,k,v,keep):
    q,k,v=map(np.asarray,(q,k,v));keep=np.asarray(keep,dtype=bool)
    if (q.dtype!=np.int32 or k.dtype!=np.int32 or v.dtype!=np.int32 or
        q.shape!=k.shape or q.shape!=v.shape or q.ndim!=2 or q.shape[1]!=64 or
        not 1<=len(q)<=512 or keep.shape!=(len(q),)):
        raise ValueError('int32 [N,64] and keep[N], 1<=N<=512 required')
    n=len(q);scores=np.empty((n,n),np.int32);ctx=np.empty_like(q);score_clips=context_clips=0
    for i in range(n):
        for j in range(n):
            dot=sum(int(a)*int(b) for a,b in zip(q[i],k[j]))
            raw=round_shift_away(dot,15);s=sat32(raw)
            score_clips+=bool(keep[j] and raw!=s)
            scores[i,j]=s if keep[j] else -(1<<31)
    probs=np.stack([softmax_fixed(row,keep) for row in scores])
    for i in range(n):
        for d in range(64):
            raw=round_shift_away(sum(int(probs[i,j])*int(v[j,d]) for j in range(n)),15)
            ctx[i,d]=sat32(raw);context_clips+=raw!=int(ctx[i,d])
    return scores,probs,ctx,{'score_saturated':int(score_clips),'context_saturated':int(context_clips)}


def attention_out_wide(x,w,b,wo,bo,keep):
    heads=w.shape[0];n,d=x.shape
    if w.shape!=(heads,3,64,d) or b.shape!=(heads,3,64):raise ValueError('bad QKV tensor shape')
    projected=np.empty((heads,3,n,64),np.int32);scores=[];probs=[];contexts=[]
    audit={'qkv_saturated':0,'score_saturated':0,'context_saturated':0}
    for h in range(heads):
        for t in range(3):
            projected[h,t],clips=linear_wide(x,w[h,t],b[h,t]);audit['qkv_saturated']+=clips
        s,p,c,report=attention_wide(*projected[h],keep)
        scores.append(s);probs.append(p);contexts.append(c)
        for k,v in report.items():audit[k]+=v
    contexts=np.stack(contexts);joined=contexts.transpose(1,0,2).reshape(n,heads*64)
    y,audit['out_projection_saturated']=linear_wide(joined,wo,bo)
    return {'projected':projected,'scores':np.stack(scores),'prob':np.stack(probs),'context':contexts,'attention':y,'audit':audit}


def transformer_wide(x,w,b,wo,bo,keep,g1,beta1,w1,b1,w2,b2,g2,beta2):
    r=attention_out_wide(x,w,b,wo,bo,keep)
    first=residual_norm_wide_fixed(x,r['attention'],g1,beta1)
    z,h,ff,fa=ffn_fixed(first['output'],w1,b1,w2,b2)
    second=residual_norm_wide_fixed(first['output'],ff,g2,beta2)
    r.update(first=first,z=z,gelu=h,ffn=ff,second=second,output=second['output'])
    r['audit'].update(first_norm_saturated=first['saturated'],second_norm_saturated=second['saturated'],**fa)
    return r
