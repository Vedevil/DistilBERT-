"""Deterministic synthetic model for code-only testing; NOT trained weights.
Numerical toy tensors match the archived Step16 synthetic regression.
"""
import numpy as np
from reference.classifier_fixed import head_fixed

def synthetic(n,d,heads,h,extreme=False):
    rng=np.random.default_rng(7000+n+d+heads+h)
    x=rng.integers(-200000,200000,(n,d),dtype=np.int32)
    w=rng.integers(-3500,3501,(heads,3,64,d),dtype=np.int16);b=rng.integers(-80000,80001,(heads,3,64),dtype=np.int32)
    wo=rng.integers(-1500,1501,(d,heads*64),dtype=np.int16);bo=rng.integers(-100000,100001,d,dtype=np.int32)
    g1=rng.integers(-20000,20001,d,dtype=np.int16);beta1=rng.integers(-100000,100001,d,dtype=np.int32)
    w1=rng.integers(-1000,1001,(h,d),dtype=np.int16);b1=rng.integers(-15000,15001,h,dtype=np.int32)
    w2=rng.integers(-1200,1201,(d,h),dtype=np.int16);b2=rng.integers(-80000,80001,d,dtype=np.int32)
    g2=rng.integers(-25000,25001,d,dtype=np.int16);beta2=rng.integers(-120000,120001,d,dtype=np.int32)
    if extreme:
        x[:,::2]=-(1<<31);x[:,1::2]=(1<<31)-1
        w[:]=32767;w[:,:,::2,:]=-32768;b[:]=2147483647;wo[:]=32767
    return x,w,b,wo,bo,g1,beta1,w1,b1,w2,b2,g2,beta2

def params_from_values(v):
    return dict(zip(['head_weight','head_bias','out_weight','out_bias','gamma','beta','ff_w1','ff_b1','ff_w2','ff_b2','gamma2','beta2'],v[1:]))

def toy_frames():
 r=np.random.default_rng(13013);D,H=3,5
 p={'pre_weight':r.integers(-32768,32768,(H,D),dtype=np.int16),'pre_bias':r.integers(-500,501,H,dtype=np.int32),'classifier_weight':r.integers(-32768,32768,(2,H),dtype=np.int16),'classifier_bias':np.array([-35,19],np.int32)}
 x=np.array([[32768,-65537,100000]],np.int32);tail=np.array([[2147483647,-2147483648,90000],[-70000,100000,44444],[131072,-262144,-2147483648]],np.int32);f=[(x,p),(np.concatenate([x,tail]),p)]
 ext={'pre_weight':np.array([[32767,-32768,32767],[-32768,32767,-32768],[32767,-32768,32767],[-32768,32767,-32768],[0,0,0]],np.int16),'pre_bias':np.zeros(H,np.int32),'classifier_weight':np.array([[-32768]*H,[32767]*H],np.int16),'classifier_bias':np.zeros(2,np.int32)}
 big=np.tile(np.array([[-2147483648,2147483647,-2147483648]],np.int32),(3,1));f.append((big,ext))
 zero={k:np.zeros_like(v) for k,v in p.items()};tie={**zero,'classifier_bias':np.array([-17,-17],np.int32)};f.append((np.zeros((2,D),np.int32),tie))
 rev={**ext,'classifier_weight':ext['classifier_weight'][::-1].copy()};f.append((big[:1],rev))
 for n in [-2147483648,2147483647]:f.append((x,{**zero,'classifier_bias':np.array([n,n],np.int32)}))
 half={**zero,'pre_weight':np.array([[4096,0,0],[-4096,0,0],[0,0,0],[0,0,0],[0,0,0]],np.int16),'classifier_weight':np.array([[4096,0,0,0,0],[-4096,0,0,0,0]],np.int16)};f.append((np.array([[1,0,0],[100000,-100000,999999]],np.int32),half))
 assert head_fixed(*f[0],max_seq=4)['prediction']==head_fixed(*f[1],max_seq=4)['prediction']
 assert np.array_equal(head_fixed(*f[0],max_seq=4)['logits'],head_fixed(*f[1],max_seq=4)['logits'])
 assert any(head_fixed(*z,max_seq=4)['pre_saturated'] for z in f)
 assert any(head_fixed(*z,max_seq=4)['classifier_saturated'] for z in f)
 assert not np.array_equal(head_fixed(*f[0],max_seq=4)['logits'],head_fixed(np.clip(x,-32768,32767).astype(np.int32),p,max_seq=4)['logits'])
 return f

def smoke_frames():
 x=None;params=[];masks=[]
 for i in range(6):
  v=list(synthetic(2,3,1,5));v[9]=v[9].astype(np.int32);v[9][0,0]=[-26097,-34866,-81670,-26097,-34866,-26097][i];v[6]=v[6]+i*173;v[12]=v[12]-i*199
  if i==0:x=v[0]
  params.append(params_from_values(v));masks.append(np.array([[1,0],[1,1],[0,0]][i%3],np.uint8))
 return x,params,masks,toy_frames()[0][1]

def evaluate(x,params,hp,mask):
    """Independent integer six-layer + classifier reference for canonical arrays."""
    from reference.layer2_fixed import transformer_layer2_fixed
    from reference.classifier_fixed import head_fixed
    keys=['head_weight','head_bias','out_weight','out_bias','gamma','beta','ff_w1','ff_b1','ff_w2','ff_b2','gamma2','beta2']
    for p in params:
        values=(x,*(p[k].astype(np.int32) if k=='ff_w2' else p[k] for k in keys))
        x=transformer_layer2_fixed(*values[:5],mask,*values[5:])['output']
    return head_fixed(x,hp)
