import tensorflow as tf


class Block(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, channels, kernel_size, dilation, cond=True):
        """Initializer.
        Args:
            channels: int, basic channel size.
            kernel_size: int, kernel size of the dilated convolution.
            dilation: int, dilation rate.
            cond: bool, whether use conditional inputs or not.
        """
        super(Block, self).__init__()
        self.channels = channels
        self.cond = cond

        self.proj_embed = tf.keras.layers.Dense(channels)
        self.conv = tf.keras.layers.Conv1D(
            channels * 2, kernel_size, padding='same', dilation_rate=dilation)

        if cond:
            self.proj_mel = tf.keras.layers.Conv1D(channels * 2, 1)

        self.proj_res = tf.keras.layers.Conv1D(channels, 1)
        self.proj_skip = tf.keras.layers.Conv1D(channels, 1)

    def call(self, inputs, embedding, mel=None):
        """Pass wavenet block.
        Args:
            inputs: tf.Tensor, [B, T, C(=channels)], input tensor.
            embedding: tf.Tensor, [B, E], embedding tensor for noise schedules.
            mel: Optional[tf.Tensor], [B, T, M], mel-spectrogram conditions.
        Returns:
            residual: tf.Tensor, [B, T, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, C], output tensor for skip connection.
        """
        # [B, C]
        embedding = self.proj_embed(embedding)
        # [B, T, C]
        x = inputs + embedding[:, None]
        # [B, T, Cx2]
        x = self.conv(x)
        if mel is not None and self.cond:
            x = x + self.proj_mel(mel)
        # [B, T, C]
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate
        # [B, T, C]
        residual = self.proj_res(x) + inputs
        skip = self.proj_skip(x)
        return residual, skip


class WaveNet(tf.keras.Model):
    """WaveNet structure.
    """
    def __init__(self, config):
        """Initializer.
        Args:
            config: Config, model configuration.
        """
        super(WaveNet, self).__init__()
        self.config = config
        # signal proj
        self.proj = tf.keras.layers.Conv1D(config.channels, 1)
        # embedding proj
        self.proj_embed = [
            tf.keras.layers.Dense(config.embedding_proj)
            for _ in range(config.embedding_layers)]
        # mel-upsampler
        if config.use_mel:
            self.upsample = [
                tf.keras.layers.Conv2DTranspose(
                    1,
                    config.upsample_kernel,
                    config.upsample_stride,
                    padding='same')
                for _ in range(config.upsample_layers)]
        # wavenet blocks
        self.blocks = []
        for i in range(config.num_layers):
            dilation = config.dilation_rate ** (i % config.num_cycles)
            self.blocks.append(
                Block(config.channels, config.kernel_size, dilation, config.use_mel))
        # [1, 1, C], initial skip block
        self.skip = tf.zeros([1, 1, config.channels])    
        # for output
        self.proj_out = [
            tf.keras.layers.Conv1D(config.channels, 1),
            tf.keras.layers.Conv1D(1, 1)]

    def call(self, signal, timestep, mel=None):
        """Generate output signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            mel: tf.Tensor, [B, T, M], mel-spectrogram.
            timestep: tf.Tensor, [B], int, timesteps of current markov chain.
        Returns:
            tf.Tensor, [B, T], generated.
        """
        # [B, T, C(=channels)]
        x = tf.nn.relu(self.proj(signal[..., None]))
        # [B, E']
        embed = self.embedding(timestep)
        # [B, E]
        for proj in self.proj_embed:
            embed = tf.nn.swish(proj(embed))
        if mel is not None and self.config.use_mel:
            # [B, T, M, 1], treat as 2D tensor.
            mel = mel[..., None]
            for upsample in self.upsample:
                mel = upsample(mel)
            # [B, T, M]
            mel = tf.squeeze(mel, axis=-1)
        # [1, 1, C]
        context = self.skip
        for block in self.blocks:
            # [B, T, C], [B, T, C]
            x, skip = block(x, embed, mel)
            # [B, T, C]
            context = context + skip
        # [B, T, 1]
        for proj in self.proj_out:
            context = proj(context)
        # [B, T]
        return tf.squeeze(context, axis=-1)

    def embedding(self, timestep):
        """Generate embedding.
        Args:
            timestep: tf.Tensor, [B], int, current timesteps.
        Returns:
            tf.Tensor, [B, E(=embedding_size)], embedding vectors.
        """
        # [E // 2]
        logit = tf.linspace(0., 1., self.config.embedding_size // 2) * \
            self.config.embedding_factor
        exp = tf.pow(10, logit)
        # [B, E // 2]
        comp = exp[None] * tf.cast(timestep[:, None], tf.float32)
        # [B, E]
        return tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)
