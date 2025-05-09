import cupy

class uniform_cache:
    def __init__(self, num: int, low: float = 0.0, high: float = 1.0):
        self._low = low
        self._high = high
        self._num_samples = num

        unif_gpu = cupy.random.uniform(low=low, high=high, size=num)
        self._cached_unif = unif_gpu.get()
        self.uniform_idx = 0
    
    def reset(self):
        unif_gpu = cupy.random.uniform(low=self._low, high=self._high, size=self._num_samples)
        self._cached_unif = unif_gpu.get()
        self.uniform_idx = 0
    
    def get_sample(self):
        s = self._cached_unif[self.uniform_idx]
        self.uniform_idx += 1
        return s