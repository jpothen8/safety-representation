"""Convert compressed trajectories once, then memory-map temporal windows."""
from contextlib import contextmanager
from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np


class TrajectoryCache:
    def __init__(self, directory, max_open=32):
        self.root=Path(directory);self.root.mkdir(parents=True,exist_ok=True)
        self.loaded=OrderedDict()
        self.max_open=max_open
        if max_open<1:raise ValueError('max_open must be positive')

    def open(self,path):
        path=Path(path).resolve();stat=path.stat()
        key=hashlib.sha256(f'{path}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()[:24]
        if key in self.loaded:
            self.loaded.move_to_end(key)
            return self.loaded[key]
        target=self.root/key
        if not (target/'manifest.json').exists():
            temporary=Path(tempfile.mkdtemp(prefix=key+'-',dir=self.root))
            try:
                manifest={}
                with np.load(path,allow_pickle=False) as source:
                    for name in source.files:
                        if '/' in name or '\\' in name:raise ValueError('Invalid trajectory key')
                        value=source[name]
                        if value.dtype.hasobject:raise ValueError('Object arrays are prohibited')
                        np.save(temporary/f'{name}.npy',value,allow_pickle=False)
                        manifest[name]=dict(shape=list(value.shape),dtype=str(value.dtype))
                (temporary/'manifest.json').write_text(json.dumps(manifest))
                try:temporary.rename(target)
                except OSError:
                    if not (target/'manifest.json').exists():raise
            finally:
                if temporary.exists():shutil.rmtree(temporary)
        manifest=json.loads((target/'manifest.json').read_text())
        arrays={name:np.load(target/f'{name}.npy',mmap_mode='r',allow_pickle=False) for name in manifest}
        self.loaded[key]=arrays
        while len(self.loaded)>self.max_open:
            _,old=self.loaded.popitem(last=False)
            for value in old.values():value._mmap.close()
        return arrays

    @contextmanager
    def trajectory(self,path):
        yield self.open(path)
