import argparse
import os
import json

import librosa
import numpy as np
import tensorflow as tf

from config import Config
from dataset import LJSpeech
from model import DiffWave

LJ_DATA_SIZE = 13100


def main(args):
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    with open(args.config) as f:
        config = Config.load(json.load(f))

    diffwave = DiffWave(config.model)
    diffwave.restore(args.ckpt).expect_partial()

    lj = LJSpeech(config.data)
    if args.offset is None:
        args.offset = config.train.split + \
            np.random.randint(LJ_DATA_SIZE - config.train.split)

    print('[*] offset: ', args.offset)
    speech = next(iter(lj.rawset.skip(args.offset)))
    speech = speech[:speech.shape[0] // config.data.hop * config.data.hop]

    librosa.output.write_wav(
        os.path.join(args.sample_dir, str(args.offset) + '_gt.wav'),
        speech.numpy(),
        config.data.sr)
    
    noise = tf.random.normal(tf.shape(speech[None]))
    librosa.output.write_wav(
        os.path.join(args.sample_dir, str(args.offset) + '_noise.wav'),
        noise[0].numpy(),
        config.data.sr)

    _, logmel = lj.mel_fn(speech[None])
    _, ir = diffwave(logmel, noise)
    for i, sample in enumerate(ir):
        librosa.output.write_wav(
            os.path.join(args.sample_dir, '{}_{}step.wav'.format(args.offset, i)),
            sample[0],
            config.data.sr)

    print('[*] done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-dir', default='./sample')
    parser.add_argument('--config', default='./ckpt/l1.json')
    parser.add_argument('--ckpt', default='./ckpt/l1/l1_500000.ckpt-1')
    parser.add_argument('--offset', default=None)
    args = parser.parse_args()
    main(args)
