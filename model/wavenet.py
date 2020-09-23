import tensorflow as tf


class Block(tf.keras.Model):
    """WaveNet Block.
    """
    def __init__(self, channels, kernel_size, dilation, last=False):
        """Initializer.
        Args:
            channels: int, basic channel size.
            kernel_size: int, kernel size of the dilated convolution.
            dilation: int, dilation rate.
            last: bool, last block or not.
        """
        super(Block, self).__init__()
        self.channels = channels
        self.last = last

        self.proj_embed = tf.keras.layers.Dense(channels)
        self.conv = tf.keras.layers.Conv1D(
            channels * 2, kernel_size, padding='same', dilation_rate=dilation)

        self.proj_mel = tf.keras.layers.Conv1D(channels * 2, 1)

        if not last:
            self.proj_res = tf.keras.layers.Conv1D(channels, 1)
        self.proj_skip = tf.keras.layers.Conv1D(channels, 1)

    def call(self, inputs, embedding, mel):
        """Pass wavenet block.
        Args:
            inputs: tf.Tensor, [B, T, C(=channels)], input tensor.
            embedding: tf.Tensor, [B, E], embedding tensor for noise schedules.
            mel: tf.Tensor, [B, T // hop, M], mel-spectrogram conditions.
        Returns:
            residual: tf.Tensor, [B, T, C], output tensor for residual connection.
            skip: tf.Tensor, [B, T, C], output tensor for skip connection.
        """
        # [B, C]
        embedding = self.proj_embed(embedding)
        # [B, T, C]
        x = inputs + embedding[:, None]
        # [B, T, Cx2]
        x = self.conv(x) + self.proj_mel(mel)
        # [B, T, C]
        context = tf.math.tanh(x[..., :self.channels])
        gate = tf.math.sigmoid(x[..., self.channels:])
        x = context * gate
        # [B, T, C]
        residual = self.proj_res(x) + inputs if not self.last else None
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
        # embedding
        self.embed = self.embedding(config.iter)
        self.proj_embed = [
            tf.keras.layers.Dense(config.embedding_proj)
            for _ in range(config.embedding_layers)]
        # mel-upsampler
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
                Block(
                    config.channels,
                    config.kernel_size,
                    dilation,
                    last=i == config.num_layers - 1))  
        # for output
        self.proj_out = [
            tf.keras.layers.Conv1D(config.channels, 1, activation=tf.nn.relu),
            tf.keras.layers.Conv1D(1, 1)]

    def call(self, signal, timestep, mel):
        """Generate output signal.
        Args:
            signal: tf.Tensor, [B, T], noised signal.
            timestep: tf.Tensor, [B], int, timesteps of current markov chain.
            mel: tf.Tensor, [B, T // hop, M], mel-spectrogram.
        Returns:
            tf.Tensor, [B, T], generated.
        """
        # [B, T, C(=channels)]
        x = tf.nn.relu(self.proj(signal[..., None]))
        # [B, E']
        embed = tf.gather(self.embed, timestep - 1)
        # [B, E]
        for proj in self.proj_embed:
            embed = tf.nn.swish(proj(embed))
        # [B, T, M, 1], treat as 2D tensor.
        mel = mel[..., None]
        for upsample in self.upsample:
            mel = tf.nn.leaky_relu(upsample(mel), self.config.leak)
        # [B, T, M]
        mel = tf.squeeze(mel, axis=-1)

        context = []
        for block in self.blocks:
            # [B, T, C], [B, T, C]
            x, skip = block(x, embed, mel)
            context.append(skip)
        # [B, T, C]
        context = tf.reduce_sum(context, axis=0)
        # [B, T, 1]
        for proj in self.proj_out:
            context = proj(context)
        # [B, T]
        return tf.squeeze(context, axis=-1)

    def embedding(self, iter):
        """Generate embedding.
        Args:
            iter: int, maximum iteration.
        Returns:
            tf.Tensor, [iter, E(=embedding_size)], embedding vectors.
        """
        # [E // 2]
        logit = tf.linspace(0., 1., self.config.embedding_size // 2)
        exp = tf.pow(10, logit * self.config.embedding_factor)
        # [iter]
        timestep = tf.range(1, iter + 1)
        # [iter, E // 2]
        comp = exp[None] * tf.cast(timestep[:, None], tf.float32)
        # [iter, E]
        return tf.concat([tf.sin(comp), tf.cos(comp)], axis=-1)
