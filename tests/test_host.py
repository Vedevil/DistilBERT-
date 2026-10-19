import gc,hashlib,json,weakref
from pathlib import Path
import numpy as np
import pytest
from host.step16.pynq_driver import DistilBert,HardwareError,DrainTimeout,InferenceTimeout,_INFLIGHT_HOLD
from host.step16.prepare_payload import convert
from reference.parameter_image import layout,pack
from reference.toy_model import smoke_frames

class Buffer(np.ndarray):
    def __new__(cls,n,addr,modern=True):
        a=np.zeros(n,dtype='<i4').view(cls);a.physical_address=addr
        if modern:a.device_address=addr
        a.freed=False;a.syncs=0;return a
    def sync_to_device(self):self.syncs+=1
    def freebuffer(self):self.freed=True
class Allocator:
    def __init__(self,modern=True):self.addr=0x1200001000;self.buffers=[];self.modern=modern
    def __call__(self,shape,dtype):
        assert dtype==np.dtype('<i4');b=Buffer(shape[0],self.addr,self.modern);self.addr+=(b.nbytes+4095)//4096*4096;self.buffers.append(b);return b
class MMIO:
    def __init__(self,mode='ok',words=6110):
        self.reg={0:DistilBert.ID,4:0x10000,0xc:8,0x10:0,0x30:words,0x34:3,0x38:5,0x3c:5,0x58:15,0x5c:16,0x64:40,0x40:(-8768)&0xffffffff,0x44:9536,0x48:1,0x4c:139430,0x50:0,0x54:7762,0x60:12232};self.writes=[];self.mode=mode;self.polls=0
    def read(self,a):
        if a==0xc and self.reg[a]&1:
            self.polls+=1
            if self.mode=='ok' and self.polls>=3:self.reg[a]=10
            elif self.mode=='error':self.reg[a]=12;self.reg[0x10]=2
        return self.reg.get(a,0)
    def write(self,a,v):
        self.writes.append((a,v))
        if a==8:
            if v==4:self.reg[0xc]=8;self.reg[0x10]=0
            elif v==1:self.reg[0xc]=17;self.polls=0
            elif v==2:
                self.reg[0x10]=4
                if self.mode!='stuck':self.reg[0xc]=12
        else:self.reg[a]=v
class Clock:
    def __init__(self):self.t=0
    def __call__(self):return self.t
    def sleep(self,s):self.t+=s
@pytest.fixture
def setup(tmp_path):
    alloc=Allocator();mmio=MMIO();clock=Clock();d=DistilBert(mmio,alloc,clock,clock.sleep)
    p=tmp_path/'params.bin';p.write_bytes(np.arange(6110,dtype='<i4').tobytes());m=layout(3,1,5,5);m['logical_signed32_SHA256']=hashlib.sha256(p.read_bytes()).hexdigest()
    return d,mmio,alloc,p,m

def load(s):s[0].load_parameters(s[3],s[4]);return s[0]
def test_signed32_high_addresses_and_consecutive(setup):
    d=load(setup);mmio=setup[1];alloc=setup[2];x=np.array([[2147483647,-2147483648,40000],[-70000,2,3]],np.int32)
    for _ in range(2):
        got=d.infer(x,[1,0]);assert got['logits_q11']==[-8768,9536] and got['label']=='POSITIVE';assert got['cycles']==139430
    assert len(alloc.buffers)==2 and alloc.buffers[0].syncs==1 and alloc.buffers[1].syncs==2
    assert (0x24,0x12) in mmio.writes and (0x2c,0x12) in mmio.writes
    assert list(alloc.buffers[1])==list(x.ravel()) and not any(b.freed for b in alloc.buffers)
    assert id(d) not in _INFLIGHT_HOLD;d.close();assert all(b.freed for b in alloc.buffers)

def test_odd_input_padding_and_resize(setup):
    d=load(setup);a=setup[2];d.infer(np.array([[1,-2,3]],np.int32),[0]);assert list(a.buffers[1])==[1,-2,3,0]
    d.infer(np.ones((4,3),np.int32),[1,1,1,1]);assert a.buffers[1].freed and len(a.buffers[2])==12;d.close()
@pytest.mark.parametrize('value',[np.zeros((2,3),np.int16),np.zeros((2,3),np.uint32),np.zeros((2,3),np.float32),np.zeros((2,3),dtype='>i4')])
def test_no_silent_narrowing_cast(value,setup):
    d=load(setup)
    with pytest.raises(TypeError):d.infer(value,[1,1])
    d.close()
@pytest.mark.parametrize('shape,mask',[((0,3),[]),((17,3),[1]*17),((2,4),[1,1]),((2,3),[1]),((2,3),[2,1]),((2,3),[1.,0.])])
def test_shape_mask_reject(shape,mask,setup):
    d=load(setup)
    with pytest.raises(ValueError):d.infer(np.zeros(shape,np.int32),mask)
    d.close()
@pytest.mark.parametrize('kind',['sha','length','dimensions','format'])
def test_bad_image_rejected(kind,setup):
    d,mmio,a,p,m=setup
    if kind=='sha':m['logical_signed32_SHA256']='0'*64
    elif kind=='length':p.write_bytes(b'0')
    elif kind=='dimensions':m['dim']=4
    else:m['format']='float32'
    with pytest.raises(ValueError):d.load_parameters(p,m)
    assert not a.buffers

def test_timeout_abort_drains_before_release(setup):
    d=load(setup);setup[1].mode='timeout'
    with pytest.raises(InferenceTimeout):d.infer(np.zeros((1,3),np.int32),[1],timeout=.003)
    assert (8,2) in setup[1].writes and id(d) not in _INFLIGHT_HOLD
    assert not any(b.freed for b in setup[2].buffers);d.close()
def test_unquiesced_timeout_pins_against_gc_reuse_and_free(setup):
    d=load(setup);mmio=setup[1];mmio.mode='stuck'
    with pytest.raises(DrainTimeout):d.infer(np.zeros((1,3),np.int32),[1],timeout=.003,drain_timeout=.002)
    assert _INFLIGHT_HOLD[id(d)] is d
    with pytest.raises(RuntimeError):d.close()
    with pytest.raises(RuntimeError):d.infer(np.zeros((1,3),np.int32),[1])
    assert not any(b.freed for b in setup[2].buffers)
    ref=weakref.ref(d);assert ref() is not None;gc.collect();assert ref() is _INFLIGHT_HOLD[id(d)]
    mmio.mode='ok';d.abort();assert id(d) not in _INFLIGHT_HOLD;d.close()

def test_hardware_error_no_logits(setup):
    d=load(setup);setup[1].mode='error'
    with pytest.raises(HardwareError,match='error 2'):d.infer(np.zeros((1,3),np.int32),[1])
    assert id(d) not in _INFLIGHT_HOLD;d.close()
def test_no_parameter_load_or_close_while_active(setup):
    d=load(setup);setup[1].reg[0xc]=17
    with pytest.raises(RuntimeError):d.load_parameters(setup[3],setup[4])
    with pytest.raises(RuntimeError):d.close()
    setup[1].reg[0xc]=8;d.close()
@pytest.mark.parametrize('address',[0x1200001004,1<<40])
def test_address_alignment_and_width(address,setup):
    d,_,a,p,m=setup;a.addr=address
    with pytest.raises(ValueError):d.load_parameters(p,m)
    assert a.buffers[0].freed

def test_physical_address_fallback_and_flush(setup):
    d=load(setup);b=d._params;del b.device_address;assert d._address(b)==b.physical_address
    class OldBuffer:
        flushed=False
        def flush(self):self.flushed=True
    b=OldBuffer();d._flush(b);assert b.flushed;d.close()
def test_hardware_id_capabilities(setup):
    _,m,a,_,_=setup;m.reg[0]=0
    with pytest.raises(ValueError):DistilBert(m,a)
    m.reg[0]=DistilBert.ID;m.reg[0x58]=0
    with pytest.raises(ValueError):DistilBert(m,a)
def test_odd_parameter_allocation(tmp_path):
    a=Allocator();m=MMIO(words=5);c=Clock();d=DistilBert(m,a,c,c.sleep);p=tmp_path/'odd.bin';p.write_bytes(np.array([-2147483648,-1,0,1,2147483647],dtype='<i4').tobytes())
    meta={'format':'Step15_signed32_parameter_words_v1','total_words':5,'dim':3,'ff_hidden':5,'head_hidden':5,'logical_signed32_SHA256':hashlib.sha256(p.read_bytes()).hexdigest()}
    d.load_parameters(p,meta);assert len(d._params)==6 and d._params[5]==0;d.close()
def test_export_binary_matches_strict_packer(tmp_path):
    x,p,m,h=smoke_frames();meta=pack(tmp_path/'params.mem',p,h);got=convert(tmp_path/'params.mem',tmp_path/'params.bin',meta)
    assert got['SHA256']==meta['logical_signed32_SHA256'] and got['words']==6110
    assert np.fromfile(tmp_path/'params.bin',dtype='<i4').min()<0
    with pytest.raises(FileExistsError):convert(tmp_path/'params.mem',tmp_path/'params.bin',meta)
@pytest.mark.parametrize('bad',['length','hash'])
def test_binary_conversion_rejects_corruption(bad,tmp_path):
    text=tmp_path/'bad.mem';text.write_text('ffffffff\n80000000\n');m={'total_words':3 if bad=='length' else 2,'logical_signed32_SHA256':'0'*64}
    with pytest.raises(ValueError):convert(text,tmp_path/'bad.bin',m)


def test_abort_requests_transport_cancel_even_with_pending_start_status(setup):
    d=load(setup);d._pin();setup[1].reg[0xc]=8;d.abort();assert setup[1].writes[-1]==(8,2);assert id(d) not in _INFLIGHT_HOLD;d.close()
@pytest.mark.parametrize('exc_type',[RuntimeError,KeyboardInterrupt])
def test_python_exception_requests_abort_and_drains_without_freeing(setup,exc_type):
    d=load(setup);mmio=setup[1];oldread=mmio.read;triggered=False
    def read(a):
        nonlocal triggered
        if a==0xc and mmio.reg[a]&1 and not triggered:triggered=True;raise exc_type('interrupted host polling')
        return oldread(a)
    mmio.read=read
    with pytest.raises(exc_type):d.infer(np.zeros((1,3),np.int32),[1])
    assert (8,2) in mmio.writes and id(d) not in _INFLIGHT_HOLD;assert not any(b.freed for b in setup[2].buffers);d.close()


def test_overlay_download_refused_with_pinned_owner_before_pynq_import(setup):
    d=load(setup);d._pin()
    with pytest.raises(RuntimeError,match='refuse overlay'):DistilBert.from_overlay('not-built.bit')
    d.abort();d.close()

def cli_fixture(setup,tmp_path,monkeypatch):
    import types,sys
    from host.step16 import pynq_driver as module
    d,mmio,alloc,p,m=setup
    (tmp_path/'parameters.bin').write_bytes(p.read_bytes());(tmp_path/'parameter_layout.json').write_text(json.dumps(m));np.save(tmp_path/'embeddings_q11.npy',np.zeros((1,3),np.int32),allow_pickle=False)
    (tmp_path/'known_cases.json').write_text(json.dumps({'mock_case':{'mask':[1],'logits_q11':[-8768,9536],'prediction':1}}))
    class Overlay:
        def __init__(self,path):self.distilbert_0=types.SimpleNamespace(mmio=mmio)
    monkeypatch.setitem(sys.modules,'pynq',types.SimpleNamespace(Overlay=Overlay,allocate=alloc));monkeypatch.setitem(sys.modules,'pynq_driver',module)
    monkeypatch.setattr(sys,'argv',['run_known_cases.py','--bit',str(tmp_path/'not-built.bit'),'--payload',str(tmp_path)])

def test_board_cli_mocked_known_case_without_pynq_or_bitstream(setup,tmp_path,monkeypatch,capsys):
    import runpy
    cli_fixture(setup,tmp_path,monkeypatch);runpy.run_path(str(Path(__file__).parents[1]/'host/step16/run_known_cases.py'),run_name='__main__')
    assert 'mock_case' in capsys.readouterr().out and all(b.freed for b in setup[2].buffers)

def test_board_cli_retains_process_on_unquiesced_exception_until_recovered(setup,tmp_path,monkeypatch,capsys):
    import runpy,time
    cli_fixture(setup,tmp_path,monkeypatch);owners=[];attempts=[];abort=DistilBert.abort
    def fail_infer(self,*a,**kw):self._pin();owners.append(self);raise DrainTimeout('mock transport fault')
    def recover(self,*a,**kw):
        attempts.append(1)
        if len(attempts)==1:raise DrainTimeout('still outstanding')
        return abort(self,*a,**kw)
    monkeypatch.setattr(DistilBert,'infer',fail_infer);monkeypatch.setattr(DistilBert,'abort',recover);monkeypatch.setattr(time,'sleep',lambda _:None)
    with pytest.raises(DrainTimeout,match='mock transport fault'):runpy.run_path(str(Path(__file__).parents[1]/'host/step16/run_known_cases.py'),run_name='__main__')
    assert len(attempts)==2 and 'KEEPING PROCESS/BUFFERS ALIVE' in capsys.readouterr().out
    assert not _INFLIGHT_HOLD and not owners[0]._active and not any(b.freed for b in setup[2].buffers);owners[0].close()

def test_generic_board_entry_point_mocked(setup,tmp_path,monkeypatch,capsys):
    import runpy
    cli_fixture(setup,tmp_path,monkeypatch);(tmp_path/'mask.json').write_text('[1]\n')
    runpy.run_path(str(Path(__file__).parents[1]/'host/step16/run_inference.py'),run_name='__main__')
    assert 'POSITIVE' in capsys.readouterr().out and all(b.freed for b in setup[2].buffers)

def test_generic_board_entry_point_retains_process_until_transport_recovers(setup,tmp_path,monkeypatch,capsys):
    import runpy,time
    cli_fixture(setup,tmp_path,monkeypatch);(tmp_path/'mask.json').write_text('[1]\n');owners=[];attempts=[];abort=DistilBert.abort
    def fail_infer(self,*a,**kw):self._pin();owners.append(self);raise DrainTimeout('mock transport fault')
    def recover(self,*a,**kw):
        attempts.append(1)
        if len(attempts)==1:raise DrainTimeout('still outstanding')
        return abort(self,*a,**kw)
    monkeypatch.setattr(DistilBert,'infer',fail_infer);monkeypatch.setattr(DistilBert,'abort',recover);monkeypatch.setattr(time,'sleep',lambda _:None)
    with pytest.raises(DrainTimeout,match='mock transport fault'):runpy.run_path(str(Path(__file__).parents[1]/'host/step16/run_inference.py'),run_name='__main__')
    assert len(attempts)==2 and 'KEEPING PROCESS/BUFFERS ALIVE' in capsys.readouterr().out and not _INFLIGHT_HOLD
    owners[0].close()
