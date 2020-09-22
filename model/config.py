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
        self.use_mel = True
        self.upsample_stride = [16, 1]
        self.upsample_kernel = [32, 3]
        self.upsample_layers = 2

        # block config
        self.channels = 64
        self.kernel_size = 3
        self.dilation_rate = 2
        self.num_layers = 30
        self.num_cycles = 3
