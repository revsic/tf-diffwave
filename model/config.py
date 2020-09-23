import numpy as np


class Config:
    """Configuration for DiffWave implementation.
    """
    def __init__(self):
        # leaky relu coefficient
        self.leak = 0.4

        # embdding config
        self.embedding_size = 128
        self.embedding_proj = 512
        self.embedding_layers = 2
        self.embedding_factor = 4

        # upsampler config
        self.upsample_stride = [16, 1]
        self.upsample_kernel = [32, 3]
        self.upsample_layers = 2
        # computed hop size
        self.hop = self.upsample_stride[0] ** self.upsample_layers

        # block config
        self.channels = 64
        self.kernel_size = 3
        self.dilation_rate = 2
        self.num_layers = 30
        self.num_cycles = 3

        # noise schedule
        self.iter = 20                  # 20, 40, 50
        self.noise_policy = 'linear'
        self.noise_start = 1e-4
        self.noise_end = 0.05           # 0.02 for 200

    def beta(self):
        """Generate beta-sequence.
        Returns:
            List[float], [iter], beta values.
        """
        mapper = {
            'linear': self._linear_sched,
        }
        if self.noise_policy not in mapper:
            raise ValueError('invalid beta policy')
        return mapper[self.noise_policy]()

    def _linear_sched(self):
        """Linearly generated noise.
        Returns:
            List[float], [iter], beta values.
        """
        return np.linspace(
            self.noise_start, self.noise_end, self.iter, dtype=np.float32)
