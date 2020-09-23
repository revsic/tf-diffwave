import tensorflow as tf


class NoamScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Noam learning rate scheduler from Vaswani et al., 2017.
    """
    def __init__(self, learning_rate, warmup_steps, channels):
        """Initializer.
        Args:
            learning_rate: float, initial learning rate.
            warmup_steps: int, warmup steps.
            channels: int, base hidden size of the model.
        """
        super(NoamScheduler, self).__init__()
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.channels = channels

    def __call__(self, step):
        """Compute learning rate.
        """
        return self.learning_rate * self.channels ** -0.5 * \
            tf.minimum(step ** -0.5, step * self.warmup_steps ** -1.5)

    def get_config(self):
        """Serialize configurations.
        """
        return {
            'learning_rate': self.learning_rate,
            'warmup_steps': self.warmup_steps,
            'channels': self.channels,
        }
