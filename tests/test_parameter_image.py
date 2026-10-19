import copy
import numpy as np
import pytest
from reference.parameter_image import layout,pack,ENC_KEYS,HEAD_KEYS
from reference.toy_model import smoke_frames

@pytest.mark.parametrize('dims',[(3,1,5,5),(768,12,3072,768),(1,1,1,1),(32,2,64,12)])
def test_word_layout_disjoint_complete(dims):
 m=layout(*dims);fields=m['encoder_fields'];end=0
 for k in ENC_KEYS:
  assert fields[k]['offset']==end;end+=fields[k]['words']
 assert end==m['encoder_words'];end*=6
 for k in HEAD_KEYS:
  assert m['head_fields'][k]['offset']==end;end+=m['head_fields'][k]['words']
 assert end==m['total_words'] and m['bytes']==end*4


@pytest.mark.parametrize('index',[0,1,2,3])
@pytest.mark.parametrize('bad',[0,4000])
def test_bad_dimension_refused(index,bad):
 v=[3,1,5,5];v[index]=bad
 with pytest.raises(ValueError):layout(*v)


def test_pack_decode_lossless_common18(tmp_path):
 _,ps,_,hp=smoke_frames();m=pack(tmp_path/'p.mem',ps,hp);raw=np.array([int(w,16) for w in (tmp_path/'p.mem').read_text().split()],np.uint32).view(np.int32);assert len(raw)==m['total_words']
 for i,p in enumerate(ps):
  for k,f in m['encoder_fields'].items():np.testing.assert_array_equal(raw[i*m['encoder_words']+f['offset']:i*m['encoder_words']+f['offset']+f['words']],p[k].ravel())
 for k,f in m['head_fields'].items():np.testing.assert_array_equal(raw[f['offset']:f['offset']+f['words']],hp[k].ravel())
 assert -81670 in raw and raw.min()<-32768
 with pytest.raises(FileExistsError):pack(tmp_path/'p.mem',ps,hp)


@pytest.mark.parametrize('kind',['float','wide_bias','wide_weight','wide_ff2','bad_shape','extra_field','few_layers','wide_gamma'])
def test_invalid_parameter_image_refused(tmp_path,kind):
 _,ps,_,hp=smoke_frames();ps=copy.deepcopy(ps)
 if kind=='float':ps[0]['gamma']=ps[0]['gamma'].astype(float)
 elif kind=='wide_bias':ps[0]['beta']=np.array([2147483648,0,0],np.int64)
 elif kind=='wide_weight':ps[0]['head_weight']=ps[0]['head_weight'].astype(np.int64);ps[0]['head_weight'][0,0,0,0]=32768
 elif kind=='wide_ff2':ps[0]['ff_w2'][0,0]=131072
 elif kind=='bad_shape':ps[0]['ff_w1']=ps[0]['ff_w1'].T.copy()
 elif kind=='extra_field':ps[0]['ignored']=np.array([1],np.int32)
 elif kind=='few_layers':ps.pop()
 else:ps[0]['gamma']=np.array([32768,0,0],np.int32)
 with pytest.raises(ValueError):pack(tmp_path/'bad.mem',ps,hp)


