"""Step16 PYNQ polling driver. Tested with MMIO/buffer doubles, NOT on a board.
Read-only DDR accelerator: flush both inputs; results are AXI-Lite registers.
Never free/reuse DMA-visible buffers until READY && !BUSY && !AXI_ACTIVE.
Unquiesced operations pin their owner globally, including after exceptions/GC.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
_INFLIGHT_HOLD={}
class HardwareError(RuntimeError):pass
class DrainTimeout(TimeoutError):pass
class InferenceTimeout(TimeoutError):pass
class DistilBert:
    ID=0x44425431
    def __init__(self,mmio,allocate,clock=time.monotonic,sleep=time.sleep):
        self.mmio=mmio;self.allocate=allocate;self.clock=clock;self.sleep=sleep
        self._params=None;self._input=None;self._active=False
        if self.read(0)!=self.ID or self.read(4)!=0x10000:raise ValueError('wrong hardware ID/version')
        self.words=self.read(0x30);self.dim=self.read(0x34);self.ff=self.read(0x38);self.hidden=self.read(0x3c);self.max_seq=self.read(0x5c);self.addr_bits=self.read(0x64)
        if not 32<=self.addr_bits<=63 or self.read(0x58)&15!=15:raise ValueError('unsupported hardware capabilities')
    @classmethod
    def from_overlay(cls,bitfile,ip_name='distilbert_0'):
        if _INFLIGHT_HOLD:raise RuntimeError('An accelerator owner is pinned/in flight; refuse overlay download until AXI is quiescent')
        from pynq import Overlay,allocate
        overlay=Overlay(str(bitfile));ip=getattr(overlay,ip_name);obj=cls(ip.mmio,allocate);obj.overlay=overlay;return obj
    def read(self,a):return int(self.mmio.read(a))&0xffffffff
    def write(self,a,v):self.mmio.write(a,int(v)&0xffffffff)
    @staticmethod
    def _address(b):
        a=getattr(b,'device_address',None)
        if a is None:a=b.physical_address
        return int(a)
    @staticmethod
    def _flush(b):
        f=getattr(b,'sync_to_device',None) or getattr(b,'flush',None)
        if f is None:raise TypeError('buffer has no cache synchronization method')
        f()
    @staticmethod
    def _free(b):
        if b is not None:b.freebuffer()
    def _quiet(self):
        s=self.read(0xc);return bool(s&8) and not bool(s&17)
    def _require_quiet(self):
        if self._active or not self._quiet():raise RuntimeError('accelerator active/draining; buffers must remain allocated')
    def _pin(self):self._active=True;_INFLIGHT_HOLD[id(self)]=self
    def _unpin(self):self._active=False;_INFLIGHT_HOLD.pop(id(self),None)
    def _wait_quiet(self,timeout):
        end=self.clock()+timeout
        while not self._quiet():
            if self.clock()>=end:raise DrainTimeout('AXI has not drained. Buffers PINNED; do not free/reload overlay. Resolve bus fault or perform a coordinated platform reset.')
            self.sleep(.001)
        self._unpin()
    def abort(self,timeout=5):
        if self._active or not self._quiet():self._pin();self.write(8,2)
        self._wait_quiet(timeout)
    def _check_buffer(self,b,words):
        a=self._address(b);size=((int(words)+1)//2)*8
        if a<0 or a%8 or a+size>1<<self.addr_bits:raise ValueError('buffer alignment/address range invalid')
        return a
    def load_parameters(self,path,manifest):
        self._require_quiet();m=json.loads(Path(manifest).read_text()) if not isinstance(manifest,dict) else manifest
        if m['format']!='Step15_signed32_parameter_words_v1' or (m['total_words'],m['dim'],m['ff_hidden'],m['head_hidden'])!=(self.words,self.dim,self.ff,self.hidden):raise ValueError('parameter image/hardware mismatch')
        path=Path(path)
        if path.stat().st_size!=self.words*4:raise ValueError('parameter byte length mismatch')
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=m['logical_signed32_SHA256']:raise ValueError('parameter SHA256 mismatch')
        b=None
        try:
            b=self.allocate(shape=((self.words+1)//2*2,),dtype=np.dtype('<i4'));self._check_buffer(b,self.words)
            src=np.memmap(path,mode='r',dtype='<i4',shape=(self.words,));b[:self.words]=src
            if len(b)>self.words:b[self.words:]=0
            self._flush(b)
        except BaseException:
            self._free(b);raise
        old=self._params;self._params=b;self._free(old)
    def infer(self,embeddings,mask,timeout=300,drain_timeout=5):
        self._require_quiet()
        if self._params is None:raise RuntimeError('load parameters first')
        a=np.asarray(embeddings)
        if a.dtype.kind!='i' or a.dtype.itemsize!=4 or a.dtype.byteorder=='>':raise TypeError('embeddings must be signed32 little/native-endian Q11, no casting/clipping')
        if a.ndim!=2 or a.shape[1]!=self.dim or not 1<=len(a)<=self.max_seq:raise ValueError('embedding dimensions')
        m=np.asarray(mask)
        if m.shape!=(len(a),) or m.dtype.kind not in 'biu' or not np.all((m==0)|(m==1)):raise ValueError('mask must contain N binary key-keep values')
        count=a.size;padded=(count+1)//2*2
        if self._input is None or len(self._input)!=padded:
            b=None
            try:b=self.allocate(shape=(padded,),dtype=np.dtype('<i4'));self._check_buffer(b,count)
            except BaseException:self._free(b);raise
            old=self._input;self._input=b;self._free(old)
        self._input[:count]=a.ravel(order='C')
        if padded>count:self._input[count:]=0
        self._flush(self._input)
        pa=self._check_buffer(self._params,self.words);ia=self._check_buffer(self._input,count)
        if pa<ia+padded*4 and ia<pa+len(self._params)*4:raise ValueError('overlapping DDR buffers')
        self.write(8,4) # clear stale DONE/ERROR before a new operation
        for base,addr in [(0x20,pa),(0x28,ia)]:self.write(base,addr);self.write(base+4,addr>>32)
        self.write(0x14,len(a));self.write(0x18,sum(int(v)<<i for i,v in enumerate(m)))
        self._pin()
        try:
            self.write(8,1);end=self.clock()+timeout
            while True:
                s=self.read(0xc)
                if s&4:
                    code=self.read(0x10);self._wait_quiet(drain_timeout);raise HardwareError(f'accelerator error {code}: 1=config,2=AXI response,3=protocol,4=abort')
                if s&2:
                    self._wait_quiet(drain_timeout)
                    raw=[self.read(0x40),self.read(0x44)];logits=[x-(1<<32) if x&(1<<31) else x for x in raw];pred=self.read(0x48)&1
                    return {'logits_q11':logits,'logits':[x/2048 for x in logits],'prediction':pred,'label':['NEGATIVE','POSITIVE'][pred],'cycles':self.read(0x4c)|(self.read(0x50)<<32),'axi_reads_low32':self.read(0x54),'word_reads_low32':self.read(0x60)}
                if self.clock()>=end:
                    self.abort(drain_timeout);raise InferenceTimeout('inference timed out; AXI successfully drained, buffers retained for reuse')
                self.sleep(.001)
        except BaseException as exc:
            # An active owner must request ABORT even if a just-issued START is
            # not yet reflected in STATUS. MMIO must preserve peripheral order.
            # An already failed drain is never declared safe based on GC/timeout.
            if self._active and not isinstance(exc,DrainTimeout):
                try:self.abort(drain_timeout)
                except BaseException as recovery:
                    if hasattr(exc,'add_note'):exc.add_note('Recovery failed; DDR ownership remains PINNED: '+str(recovery))
            raise
    def close(self):
        self._require_quiet();self._free(self._input);self._free(self._params);self._input=None;self._params=None
