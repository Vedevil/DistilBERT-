"""Version1 word-addressed read-only parameter layout. No new quantization.
All scalar words signed32; native16/17/18 weights losslessly sign extended.
Actual packed image (~165MiB) belongs ONLY in excluded cache, never snapshots.
"""
import hashlib,json
from pathlib import Path
import numpy as np
ENC_KEYS=['gamma2','beta2','gamma','beta','head_bias','head_weight','out_bias','out_weight','ff_b1','ff_w1','ff_b2','ff_w2']
HEAD_KEYS=['pre_bias','pre_weight','classifier_bias','classifier_weight']
def layout(d=768,heads=12,ff=3072,head_hidden=768):
 if not 1<=d<=3072 or not 1<=heads<=12 or not 1<=ff<=3072 or not 1<=head_hidden<=3072:raise ValueError('unsupported dimensions')
 sizes=[d,d,d,d,heads*192,heads*192*d,d,d*heads*64,ff,ff*d,d,d*ff];offset=0;fields={}
 for k,n in zip(ENC_KEYS,sizes):fields[k]={'offset':offset,'words':n};offset+=n
 enc=offset;hsizes=[head_hidden,head_hidden*d,2,2*head_hidden];hfields={};offset=6*enc
 for k,n in zip(HEAD_KEYS,hsizes):hfields[k]={'offset':offset,'words':n};offset+=n
 return {'format':'Step15_signed32_parameter_words_v1','address_unit':'32bit_WORD_not_byte','layers':6,'dim':d,'heads':heads,'ff_hidden':ff,'head_hidden':head_hidden,'encoder_words':enc,'encoder_fields':fields,'head_fields':hfields,'total_words':offset,'bytes':4*offset,'weight_scaling':'unchangedQ13_signed16_or_common18,signextended_to32','bias_gamma_scaling':'biasQ11,gammaQ13,unchanged','memory_reset_contract':'The logical core response stream must discard aborted replies. AXI soft abort preserves ARVALID and drains physical traffic before READY/restart; cold reset requires fabric quiescence/reset.'}
def pack(path,params,hp):
 path=Path(path)
 if path.exists():raise FileExistsError(path)
 if len(params)!=6:raise ValueError('six encoder parameter sets required')
 d=params[0]['out_bias'].size;heads=params[0]['head_weight'].shape[0];ff=params[0]['ff_b1'].size;hh=hp['pre_bias'].size;m=layout(d,heads,ff,hh);h=hashlib.sha256();count=0
 with path.open('x') as f:
  for p,keys,fields in [(p,ENC_KEYS,m['encoder_fields']) for p in params]+[(hp,HEAD_KEYS,m['head_fields'])]:
   if set(p)!=set(keys):raise ValueError('parameter field schema')
   eshapes={'gamma2':(d,),'beta2':(d,),'gamma':(d,),'beta':(d,),'head_bias':(heads,3,64),'head_weight':(heads,3,64,d),'out_bias':(d,),'out_weight':(d,heads*64),'ff_b1':(ff,),'ff_w1':(ff,d),'ff_b2':(d,),'ff_w2':(d,ff),'pre_bias':(hh,),'pre_weight':(hh,d),'classifier_bias':(2,),'classifier_weight':(2,hh)}
   for k in keys:
    if np.shape(p[k])!=eshapes[k]:raise ValueError('parameter shape '+k)
    a=np.asarray(p[k]);n=fields[k]['words']
    if a.size!=n or not np.issubdtype(a.dtype,np.signedinteger) or np.any(a.astype(np.int64)<-2147483648) or np.any(a.astype(np.int64)>2147483647):raise ValueError('parameter schema/range '+k)
    if 'weight' in k or k in ['ff_w1','ff_w2','gamma','gamma2']:
     bound=131072 if k=='ff_w2' else 32768
     if np.any(a.astype(np.int64)<-bound) or np.any(a.astype(np.int64)>=bound):raise ValueError('weight format overflow')
    v=a.astype('<i4').ravel();h.update(v.tobytes());count+=len(v)
    for start in range(0,len(v),65536):f.write(''.join(f'{int(x)&0xffffffff:08x}\n' for x in v[start:start+65536]))
 assert count==m['total_words'];m['logical_signed32_SHA256']=h.hexdigest();m['text_bytes']=path.stat().st_size;return m
