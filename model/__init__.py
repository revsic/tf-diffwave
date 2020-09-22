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

    def call(self, noise, mel=None):
        """Generate denoised audio.
        Args:
            noise: tf.Tensor, [B, T], starting noise.
            mel: Optional[tf.Tensor], [B, T // hop, M], conditonal mel-spectrogram.
        Returns:
            tuple,
                signal: tf.Tensor, [B, T], predicted output.
                ir: List[np.ndarray: [B, T]], intermediate outputs.
        """
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
            tf.Tensor, [B, T], noised signal.
        """
        if eps is None:
            eps = tf.random.normal(tf.shape(signal))
        return tf.sqrt(alpha_bar) * signal + tf.sqrt(1 - alpha_bar) * eps

    def pred_noise(self, signal, timestep, mel=None):
        """Predict noise from signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], timesteps of current markov chain.
            mel: Optional[tf.Tensor], [B, T // hop, M], conditional mel-spectrogram.
        Returns:
            tf.Tensor, [B, T], predicted noise.
        """
        return self.wavenet(signal, timestep, mel)

    def pred_signal(self, signal, eps, alpha, alpha_bar):
        """Compute mean and stddev of denoised signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            eps: tf.Tensor, [B, T], estimated noise.
            alpha: Union[float, tf.Tensor: [B]], 1 - beta.
            alpha_bar: Union[float, tf.Tensor: [B]], cumprod(1 - beta).
        Returns:
            tuple,
                mean: tf.Tensor, [B, T], estimated mean of denoised signal.
                stddev: float, estimated stddev.
        """
        # [B, T]
        mean = (signal - (1 - alpha) / tf.sqrt(1 - alpha_bar) * eps) / tf.sqrt(alpha)
        # []
        stddev = tf.sqrt((1 - alpha_bar / alpha) / (1 - alpha_bar) * (1 - alpha))
        return mean, stddev
