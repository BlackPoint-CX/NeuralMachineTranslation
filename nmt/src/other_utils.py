import math

import tensorflow as tf
from tensorflow import clip_by_global_norm


def get_max_time(tensor, time_major=False):
    time_axis = 0 if time_major else 1
    return tensor.shape[time_axis].value or tf.shape(tensor)[time_axis]


def gradient_clip(gradients, max_gradient_norm):
    clipped_gradients, gradient_norm = clip_by_global_norm(gradients, max_gradient_norm)

    gradient_norm_summary = [tf.summary.scalar('grad_norm', gradient_norm),
                             tf.summary.scalar('clipped_gradient', tf.global_norm(clipped_gradients))]

    return clipped_gradients, gradient_norm_summary, gradient_norm



def safe_exp(value):
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float('inf')

    return ans
