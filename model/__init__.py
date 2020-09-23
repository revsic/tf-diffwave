import numpy as np
import tensorflow as tf

from .wavenet import WaveNet


class DiffWave(tf.keras.Model):
    """DiffWave: A Versatile Diffusion Model for Audio Synthesis.
    Zhifeng Kong et al., 2020.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(DiffWave, self).__init__()
        self.config = config
        self.wavenet = WaveNet(config)

    def call(self, mel, noise=None):
        """Generate denoised audio.
        Args:
            mel: tf.Tensor, [B, T // hop, M], conditonal mel-spectrogram.
            noise: Optional[tf.Tensor], [B, T], starting noise.
        Returns:
            tuple,
                signal: tf.Tensor, [B, T], predicted output.
                ir: List[np.ndarray: [B, T]], intermediate outputs.
        """
        if noise is None:
            # [B, T // hop, M]
            b, t, _ = tf.shape(mel)
            # [B, T]
            noise = tf.random.normal([b, t * self.config.hop])

        # [iter]
        alpha = 1 - self.config.beta()
        alpha_bar = np.cumprod(alpha)
        # [B]
        base = tf.ones([tf.shape(noise)[0]])

        ir, signal = [], noise
        for t in range(self.config.iter, 0, -1):
            # [B, T]
            eps = self.pred_noise(signal, base * t, mel)
            # [B, T], []
            mu, sigma = self.pred_signal(signal, eps, alpha[t - 1], alpha_bar[t - 1])
            # [B, T]
            signal = mu + tf.random.normal(tf.shape(signal)) * sigma
            ir.append(signal.numpy())
        # [B, T], iter x [B, T]
        return signal, ir

    def diffusion(self, signal, alpha_bar, eps=None):
        """Trans to next state with diffusion process.
        Args:
            signal: tf.Tensor, [B, T], signal.
            alpha_bar: Union[float, tf.Tensor: [B]], cumprod(1 -beta).
            eps: Optional[tf.Tensor: [B, T]], noise.
        Return:
            tuple,
                noised: tf.Tensor, [B, T], noised signal.
                eps: tf.Tensor, [B, T], noise.
        """
        if eps is None:
            eps = tf.random.normal(tf.shape(signal))
        if isinstance(alpha_bar, tf.Tensor):
            alpha_bar = alpha_bar[:, None]
        return tf.sqrt(alpha_bar) * signal + tf.sqrt(1 - alpha_bar) * eps, eps

    def pred_noise(self, signal, timestep, mel):
        """Predict noise from signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], timesteps of current markov chain.
            mel: tf.Tensor, [B, T // hop, M], conditional mel-spectrogram.
        Returns:
            tf.Tensor, [B, T], predicted noise.
        """
        return self.wavenet(signal, timestep, mel)

    def pred_signal(self, signal, eps, alpha, alpha_bar):
        """Compute mean and stddev of denoised signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            eps: tf.Tensor, [B, T], estimated noise.
            alpha: float, 1 - beta.
            alpha_bar: float, cumprod(1 - beta).
        Returns:
            tuple,
                mean: tf.Tensor, [B, T], estimated mean of denoised signal.
                stddev: float, estimated stddev.
        """
        # [B, T]
        mean = (signal - (1 - alpha) / np.sqrt(1 - alpha_bar) * eps) / np.sqrt(alpha)
        # []
        stddev = np.sqrt((1 - alpha_bar / alpha) / (1 - alpha_bar) * (1 - alpha))
        return mean, stddev

    def write(self, path, optim=None):
        """Write checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to write.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.save(path)

    def restore(self, path, optim=None):
        """Restore checkpoint with `tf.train.Checkpoint`.
        Args:
            path: str, path to restore.
            optim: Optional[tf.keras.optimizers.Optimizer]
                , optional optimizer.
        """
        kwargs = {'model': self}
        if optim is not None:
            kwargs['optim'] = optim
        ckpt = tf.train.Checkpoint(**kwargs)
        ckpt.restore(path)
