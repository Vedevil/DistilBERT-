#!/usr/bin/env python3
"""Integrity check for the prepared source snapshot, not a ban on later edits."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parents[1];m=json.loads((root/'SOURCE_MANIFEST.json').read_text());bad=[]
for rel,meta in m['files'].items():
    p=root/rel
    if not p.is_file() or p.stat().st_size!=meta['bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=meta['sha256']:bad.append(rel)
if bad:print('Changed/missing files:',*bad,sep='\n');sys.exit(1)
print('PASS source snapshot:',len(m['files']),'files, SHA256/size match (manifest excludes itself).')
