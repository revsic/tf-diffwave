import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tqdm

from config import Config
from dataset import LJSpeech
from model import DiffWave


class Trainer:
    """WaveGrad trainer.
    """
    def __init__(self, model, lj, config):
        """Initializer.
        Args:
            model: DiffWave, diffwave model.
            lj: LJSpeech, LJ-speec dataset
                which provides already batched and normalized speech dataset.
            config: Config, unified configurations.
        """
        self.model = model
        self.lj = lj
        self.config = config

        self.split = config.train.split // config.data.batch
        self.trainset = self.lj.dataset().take(self.split) \
            .shuffle(config.train.bufsiz) \
            .prefetch(tf.data.experimental.AUTOTUNE)
        self.testset = self.lj.dataset().skip(self.split) \
            .prefetch(tf.data.experimental.AUTOTUNE)

        self.optim = tf.keras.optimizers.Adam(
            config.train.lr(),
            config.train.beta1,
            config.train.beta2,
            config.train.eps)

        self.eval_intval = config.train.eval_intval // config.data.batch
        self.ckpt_intval = config.train.ckpt_intval // config.data.batch

        self.train_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'train'))
        self.test_log = tf.summary.create_file_writer(
            os.path.join(config.train.log, config.train.name, 'test'))

        self.ckpt_path = os.path.join(
            config.train.ckpt, config.train.name, config.train.name)

        self.alpha_bar = np.cumprod(1 - config.model.beta())
        self.cmap = tf.constant(plt.get_cmap('viridis').colors, dtype=tf.float32)

    def compute_loss(self, signal, logmel):
        """Compute loss for noise estimation.
        Args:
            signal: tf.Tensor, [B, T], raw audio signal segment.
            logmel: tf.Tensor, [B, T // hop, mel], mel-spectrogram.
        Returns:
            loss: tf.Tensor, [], L1-loss between noise and estimation.
        """
        # [B]
        bsize = tf.shape(signal)[0]
        # [B]
        timesteps = tf.random.uniform(
            [bsize], 1, self.config.model.iter + 1, dtype=tf.int32)
        # [B]
        noise_level = tf.gather(self.alpha_bar, timesteps - 1)
        # [B, T], [B, T]
        noised, noise = self.model.diffusion(signal, noise_level)
        # [B, T]
        eps = self.model.pred_noise(noised, timesteps, logmel)
        # []
        loss = tf.reduce_mean(tf.abs(eps - noise))
        return loss

    def train(self, step=0, ir_unit=5):
        """Train wavegrad.
        Args:
            step: int, starting step.
            ir_unit: int, log ir units.
        """
        for _ in tqdm.trange(step // self.split, self.config.train.epoch):
            with tqdm.tqdm(total=self.split, leave=False) as pbar:
                for signal, logmel in self.trainset:
                    with tf.GradientTape() as tape:
                        tape.watch(self.model.trainable_variables)
                        loss = self.compute_loss(signal, logmel)

                    grad = tape.gradient(loss, self.model.trainable_variables)
                    self.optim.apply_gradients(
                        zip(grad, self.model.trainable_variables))

                    norm = tf.reduce_mean([tf.norm(g) for g in grad])
                    del grad

                    step += 1
                    pbar.update()
                    pbar.set_postfix(
                        {'loss': loss.numpy().item(),
                         'step': step,
                         'grad': norm.numpy().item()})

                    with self.train_log.as_default():
                        tf.summary.scalar('loss', loss, step)
                        tf.summary.scalar('grad norm', norm, step)
                        if step % self.eval_intval == 0:
                            pred, _ = self.model(logmel)
                            tf.summary.audio(
                                'train', pred[..., None], self.config.data.sr, step)
                            tf.summary.image(
                                'train mel', self.mel_img(pred), step)

                            del pred

                    if step % self.ckpt_intval == 0:
                        self.model.write(
                            '{}_{}.ckpt'.format(self.ckpt_path, step),
                            self.optim)

            loss = [
                self.compute_loss(signal, logmel).numpy().item()
                for signal, logmel in self.testset
            ]
            loss = sum(loss) / len(loss)
            with self.test_log.as_default():
                tf.summary.scalar('loss', loss, step)

                gt, pred, ir = self.eval()
                tf.summary.audio(
                    'gt', gt[None, :, None], self.config.data.sr, step)
                tf.summary.audio(
                    'eval', pred[None, :, None], self.config.data.sr, step)

                tf.summary.image(
                    'gt mel', self.mel_img(gt[None]), step)
                tf.summary.image(
                    'eval mel', self.mel_img(pred[None]), step)

                for i in range(0, len(ir), ir_unit):
                    tf.summary.audio(
                        'ir_{}'.format(i),
                        np.clip(ir[i][None, :, None], -1., 1.),
                        self.config.data.sr, step)
                
                del gt, pred, ir

    def mel_img(self, signal):
        """Generate mel-spectrogram images.
        Args:
            signal: tf.Tensor, [B, T], speech signal.
        Returns:
            tf.Tensor, [B, mel, T // hop, 3], mel-spectrogram in viridis color map.
        """
        # [B, T // hop, mel]
        _, mel = self.lj.mel_fn(signal)
        # [B, mel, T // hop]
        mel = tf.transpose(mel, [0, 2, 1])
        # minmax norm in range(0, 1)
        mel = (mel - tf.reduce_min(mel)) / (tf.reduce_max(mel) - tf.reduce_min(mel))
        # in range(0, 255)
        mel = tf.cast(mel * 255, tf.int32)
        # [B, mel, T // hop, 3]
        mel = tf.gather(self.cmap, mel)
        # make origin lower
        mel = tf.image.flip_up_down(mel)
        return mel

    def eval(self):
        """Generate evaluation purpose audio.
        Returns:
            speech: np.ndarray, [T], ground truth.
            pred: np.ndarray, [T], predicted.
            ir: List[np.ndarray], config.model.iter x [T],
                intermediate represnetations.
        """
        # [T]
        speech = next(iter(lj.rawset))
        # [1, T // hop, mel]
        _, logmel = lj.mel_fn(speech[None])
        # [1, T], iter x [1, T]
        pred, ir = self.model(logmel)
        # [T]
        pred = tf.squeeze(pred, axis=0).numpy()
        # config.model.iter x [T]
        ir = [np.squeeze(i, axis=0) for i in ir]
        return speech.numpy(), pred, ir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None)
    parser.add_argument('--load-step', default=0, type=int)
    parser.add_argument('--ir-unit', default=10, type=int)
    parser.add_argument('--data-dir', default=None)
    parser.add_argument('--download', default=False, action='store_true')
    parser.add_argument('--from-raw', default=False, action='store_true')
    args = parser.parse_args()

    config = Config()
    if args.config is not None:
        print('[*] load config: ' + args.config)
        with open(args.config) as f:
            config = Config.load(json.load(f))

    log_path = os.path.join(config.train.log, config.train.name)
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    ckpt_path = os.path.join(config.train.ckpt, config.train.name)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    lj = LJSpeech(config.data, args.data_dir, args.download, not args.from_raw)
    diffwave = DiffWave(config.model)
    trainer = Trainer(diffwave, lj, config)

    if args.load_step > 0:
        super_path = os.path.join(config.train.ckpt, config.train.name)
        ckpt_path = '{}_{}.ckpt'.format(config.train.name, args.load_step)
        ckpt_path = next(
            name for name in os.listdir(super_path)
                 if name.startswith(ckpt_path) and name.endswith('.index'))
        ckpt_path = os.path.join(super_path, ckpt_path[:-6])
        
        print('[*] load checkpoint: ' + ckpt_path)
        trainer.model.restore(ckpt_path, trainer.optim)

    with open(os.path.join(config.train.ckpt, config.train.name + '.json'), 'w') as f:
        json.dump(config.dump(), f)

    trainer.train(args.load_step, args.ir_unit)
