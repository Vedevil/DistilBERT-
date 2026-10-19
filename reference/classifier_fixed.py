"""Step13 independent integer head specification; no HDL-derived expected data.
Full int32Q11 token rows; select CLS row0, linear→ReLU→linear→two logits.
Dropout disabled, no softmax needed for argmax. Ties select class0 NEGATIVE.
"""
import numpy as np
from reference.ffn_fixed import linear_wide

KEYS={'pre_weight','pre_bias','classifier_weight','classifier_bias'}
def head_fixed(sequence,p,max_seq=16):
    x=np.asarray(sequence)
    if x.dtype!=np.int32 or x.ndim!=2 or not 1<=x.shape[0]<=max_seq:raise ValueError('int32Q11 rows and legal nonempty length required')
    if set(p)!=KEYS:raise ValueError('wrong head tensor set')
    w,b,c,d=(p[k] for k in ['pre_weight','pre_bias','classifier_weight','classifier_bias'])
    if w.ndim!=2 or not 1<=w.shape[0]<=3072 or c.shape!=(2,w.shape[0]) or d.shape!=(2,):raise ValueError('wrong head dimensions')
    pre,sc1=linear_wide(x[:1],w,b);relu=np.maximum(pre,0).astype(np.int32)
    logits,sc2=linear_wide(relu,c,d)
    return {'cls':x[0].copy(),'pre':pre[0],'relu':relu[0],'logits':logits[0],
        'prediction':int(np.argmax(logits[0])),'pre_saturated':sc1,'classifier_saturated':sc2}
