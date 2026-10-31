import zipfile, pickle, collections
import numpy as np

class _Marker:
    def __init__(self, name): self.__name__ = name
class _Dummy:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return _Dummy()
    def __setstate__(self, s): pass

_DTYPES = {'FloatStorage': np.float32, 'DoubleStorage': np.float64, 'LongStorage': np.int64,
           'IntStorage': np.int32, 'HalfStorage': np.float16, 'BoolStorage': np.bool_}

def load_checkpoint_numpy(path):
    zf = zipfile.ZipFile(path)
    names = zf.namelist()
    prefix = [n for n in names if n.endswith('data.pkl')][0].rsplit('/', 1)[0]
    def rebuild(storage, offset, size, stride, *rest):
        size = tuple(size); stride = tuple(stride)
        item = storage.itemsize
        arr = np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                              strides=tuple(s * item for s in stride))
        return np.array(arr)
    class U(pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'collections' and name == 'OrderedDict': return collections.OrderedDict
            if module == 'torch._utils' and name == '_rebuild_tensor_v2': return rebuild
            if module == 'torch' and name.endswith('Storage'): return _Marker(name)
            if module == 'torch' and name == 'Size': return tuple
            return _Dummy
        def persistent_load(self, pid):
            _, stype, key, _loc, _n = pid
            dt = _DTYPES[stype.__name__]
            return np.frombuffer(zf.read(f'{prefix}/data/{key}'), dtype=dt)
    return U(zf.open(f'{prefix}/data.pkl')).load()

class NumpyAutopilot:
    """Pure-numpy forward pass of AutopilotNet (eval mode). Output: [velocity, steering]."""
    def __init__(self, path):
        ck = load_checkpoint_numpy(path)
        self.epoch = ck.get('epoch')
        self.sd = {k: np.asarray(v, dtype=np.float32) for k, v in ck['state_dict'].items()}
    @staticmethod
    def _conv(x, w, b):
        import cv2
        co, ci, kh, kw = w.shape
        H, W = x.shape[1:]
        out = np.zeros((co, H - kh + 1, W - kw + 1), np.float32)
        for o in range(co):
            acc = np.zeros_like(out[o])
            for i in range(ci):
                # cv2.filter2D computes correlation, same as torch conv
                full = cv2.filter2D(x[i], -1, w[o, i], anchor=(kw // 2, kh // 2), borderType=cv2.BORDER_CONSTANT)
                acc += full[kh // 2: kh // 2 + out.shape[1], kw // 2: kw // 2 + out.shape[2]]
            out[o] = acc + b[o]
        return out
    @staticmethod
    def _pool(x):
        C, H, W = x.shape
        x = x[:, :H // 2 * 2, :W // 2 * 2].reshape(C, H // 2, 2, W // 2, 2)
        return x.max(axis=(2, 4))
    def forward(self, img):
        """img: float32 array (96,128) already normalised to [-1,1]."""
        s = self.sd
        x = img[None].astype(np.float32)
        x = np.maximum(self._conv(x, s['convolutional_layers.0.weight'], s['convolutional_layers.0.bias']), 0)
        x = self._pool(x)
        x = np.maximum(self._conv(x, s['convolutional_layers.3.weight'], s['convolutional_layers.3.bias']), 0)
        x = self._pool(x).reshape(-1)
        x = np.maximum(s['shared_layers.0.weight'] @ x + s['shared_layers.0.bias'], 0)
        def head(p):
            h = np.maximum(s[f'{p}.0.weight'] @ x + s[f'{p}.0.bias'], 0)
            return float((s[f'{p}.2.weight'] @ h + s[f'{p}.2.bias'])[0])
        return head('velocity_head'), head('steering_head')

if __name__ == '__main__':
    ck = load_checkpoint_numpy('/mnt/user-data/uploads/model.pt')
    print('keys:', list(ck.keys()), 'epoch:', ck.get('epoch'))
    for k, v in ck['state_dict'].items():
        print(f'{k:42s} {tuple(v.shape)} mean={v.mean():+.4f} std={v.std():.4f}')
    print('scheduler:', {k: v for k, v in ck.get('scheduler', {}).items() if isinstance(v, (int, float, str))})
