# Import TF 2.X and make sure we're running eager.
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()
assert tf.executing_eagerly()

import tensorflow_hub as hub
import numpy as np

# Load the module and run inference.
module = hub.load('https://tfhub.dev/google/nonsemantic-speech-benchmark/trill-distilled/3')
# `wav_as_float_or_int16` can be a numpy array or tf.Tensor of float type or
# int16. The sample rate must be 16kHz. Resample to this sample rate, if
# necessary.
wav_as_float_or_int16 = np.sin(np.linspace(-np.pi, np.pi, 128), dtype=np.float32)
emb = module(samples=wav_as_float_or_int16, sample_rate=16000)['embedding']
# `emb` is a [time, feature_dim] Tensor.
emb.shape.assert_is_compatible_with([None, 2048])

print(emb)

