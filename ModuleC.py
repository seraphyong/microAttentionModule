from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sonnet as snt
import tensorflow as tf
import math
import os
import cv2
import numpy as np


class Unit3D(snt.AbstractModule):
    """Basic unit containing Conv3D + BatchNorm + non-linearity."""

    def __init__(self, output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 activation_fn=tf.nn.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__(name=name)
        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias

    def _build(self, inputs, is_training):
        """Connects the module to inputs.

        Args:
          inputs: Inputs to the Unit3D component.
          is_training: whether to use training mode for snt.BatchNorm (boolean).

        Returns:
          Outputs from the module.
        """
        net = snt.Conv3D(output_channels=self._output_channels,
                         kernel_shape=self._kernel_shape,
                         stride=self._stride,
                         padding=snt.SAME,
                         use_bias=self._use_bias)(inputs)
        if self._use_batch_norm:
            bn = snt.BatchNorm()
            net = bn(net, is_training=is_training, test_local_stats=False)
        if self._activation_fn is not None:
            net = self._activation_fn(net)
        return net


class InceptionI3d(snt.AbstractModule):
    """Inception-v1 I3D architecture.

    The model is introduced in:

      Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset
      Joao Carreira, Andrew Zisserman
      https://arxiv.org/pdf/1705.07750v1.pdf.

    See also the Inception architecture, introduced in:

      Going deeper with convolutions
      Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
      Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.
      http://arxiv.org/pdf/1409.4842v1.pdf.
    """

    # Endpoints of the model in order. During construction, all the endpoints up
    # to a designated `final_endpoint` are returned in a dictionary as the
    # second return value.
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
        'Logits',
        'Predictions',
    )

    def __init__(self, num_classes=400, spatial_squeeze=True,
                 final_endpoint='Logits', name='inception_i3d'):
        """Initializes I3D model instance.

        Args:
          num_classes: The number of outputs in the logit layer (default 400, which
              matches the Kinetics dataset).
          spatial_squeeze: Whether to squeeze the spatial dimensions for the logits
              before returning (default True).
          final_endpoint: The model contains many possible endpoints.
              `final_endpoint` specifies the last endpoint for the model to be built
              up to. In addition to the output at `final_endpoint`, all the outputs
              at endpoints up to `final_endpoint` will also be returned, in a
              dictionary. `final_endpoint` must be one of
              InceptionI3d.VALID_ENDPOINTS (default 'Logits').
          name: A string (optional). The name of this module.

        Raises:
          ValueError: if `final_endpoint` is not recognized.
        """

        if final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % final_endpoint)

        super(InceptionI3d, self).__init__(name=name)
        self._num_classes = num_classes
        self._spatial_squeeze = spatial_squeeze
        self._final_endpoint = final_endpoint

    def _build(self, inputs, is_training, dropout_keep_prob=1.0):
        """Connects the model to inputs.

        Args:
          inputs: Inputs to the model, which should have dimensions
              `batch_size` x `num_frames` x 224 x 224 x `num_channels`.
          is_training: whether to use training mode for snt.BatchNorm (boolean).
          dropout_keep_prob: Probability for the tf.nn.dropout layer (float in
              [0, 1)).

        Returns:
          A tuple consisting of:
            1. Network output at location `self._final_endpoint`.
            2. Dictionary containing all endpoints up to `self._final_endpoint`,
               indexed by endpoint name.

        Raises:
          ValueError: if `self._final_endpoint` is not recognized.
        """
        if self._final_endpoint not in self.VALID_ENDPOINTS:
            raise ValueError('Unknown final endpoint %s' % self._final_endpoint)

        net = inputs
        end_points = {}
        end_point = 'Conv3d_1a_7x7'
        net = Unit3D(output_channels=64, kernel_shape=[7, 7, 7],
                     stride=[2, 2, 2], name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        # print(net.shape)
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool3d_2a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net

        if self._final_endpoint == end_point: return net, end_points
        end_point = 'Conv3d_2b_1x1'
        net = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                     name=end_point)(net, is_training=is_training)
        end_points[end_point] = net

        if self._final_endpoint == end_point: return net, end_points
        end_point = 'Conv3d_2c_3x3'
        net = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                     name=end_point)(net, is_training=is_training)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points
        end_point = 'MaxPool3d_3a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 1, 3, 3, 1], strides=[1, 1, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)
                branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)
                branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=32, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)

            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_3c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)
                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=192, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)
                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=96, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_4a_3x3'
        net = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1], strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)
                branch_1 = Unit3D(output_channels=96, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=208, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=208, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)
                branch_2 = Unit3D(output_channels=16, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=48, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)

                branch_1 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=224, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=224, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)

                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4d'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)

                branch_1 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=256, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)

                branch_2 = Unit3D(output_channels=24, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4e'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=112, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=144, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)

                branch_1 = Unit3D(output_channels=114, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=288, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=288, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)

                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=64, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=64, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_4f'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)

                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=320, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)

                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'MaxPool3d_5a_2x2'
        net = tf.nn.max_pool3d(net, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1],
                               padding=snt.SAME, name=end_point)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5b'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=256, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)

                branch_1 = Unit3D(output_channels=160, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=320, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=320, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0a_3x3')(branch_2,
                #                                         is_training=is_training)

                branch_2 = Unit3D(output_channels=32, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Mixed_5c'
        with tf.variable_scope(end_point):
            with tf.variable_scope('Branch_0'):
                branch_0 = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
            with tf.variable_scope('Branch_1'):
                # branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_1 = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_1,
                #                                         is_training=is_training)

                branch_1 = Unit3D(output_channels=192, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_1_a = Unit3D(output_channels=384, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_1,
                                                          is_training=is_training)
                branch_1_b = tf.nn.max_pool3d(branch_1_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_1a_3x3')
                # normal module
                branch_1_c = Unit3D(output_channels=384, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_1,
                                                          is_training=is_training)
                branch_1 = branch_1_b + branch_1_c
            with tf.variable_scope('Branch_2'):
                # branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                #                   name='Conv3d_0a_1x1')(net, is_training=is_training)
                # branch_2 = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                #                   name='Conv3d_0b_3x3')(branch_2,
                #                                         is_training=is_training)

                branch_2 = Unit3D(output_channels=48, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0a_1x1')(net, is_training=is_training)
                # add att module
                branch_2_a = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                    name='Conv3d_0b_1x1')(branch_2,
                                                          is_training=is_training)
                branch_2_b = tf.nn.max_pool3d(branch_2_a, ksize=[1, 3, 3, 3, 1],
                                              strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                              name='MaxPool3d_2a_3x3')
                # normal module
                branch_2_c = Unit3D(output_channels=128, kernel_shape=[3, 3, 3],
                                    name='Conv3d_0b_3x3')(branch_2,
                                                          is_training=is_training)
                branch_2 = branch_2_b + branch_2_c
            with tf.variable_scope('Branch_3'):
                branch_3 = tf.nn.max_pool3d(net, ksize=[1, 3, 3, 3, 1],
                                            strides=[1, 1, 1, 1, 1], padding=snt.SAME,
                                            name='MaxPool3d_0a_3x3')
                branch_3 = Unit3D(output_channels=128, kernel_shape=[1, 1, 1],
                                  name='Conv3d_0b_1x1')(branch_3,
                                                        is_training=is_training)
            net = tf.concat([branch_0, branch_1, branch_2, branch_3], 4)
        end_points[end_point] = net
        if self._final_endpoint == end_point: return net, end_points

        end_point = 'Logits'
        with tf.variable_scope(end_point):
            net = tf.nn.avg_pool3d(net, ksize=[1, 2, 7, 7, 1],
                                   strides=[1, 1, 1, 1, 1], padding=snt.VALID)
            net = tf.nn.dropout(net, dropout_keep_prob)
            logits = Unit3D(output_channels=self._num_classes,
                            kernel_shape=[1, 1, 1],
                            activation_fn=None,
                            use_batch_norm=False,
                            use_bias=True,
                            name='Conv3d_0c_1x1')(net, is_training=is_training)
            if self._spatial_squeeze:
                logits = tf.squeeze(logits, [2, 3], name='SpatialSqueeze')
        averaged_logits = tf.reduce_mean(logits, axis=1)
        end_points[end_point] = averaged_logits
        if self._final_endpoint == end_point: return averaged_logits, end_points

        end_point = 'Predictions'
        predictions = tf.nn.softmax(averaged_logits)
        end_points[end_point] = predictions
        return predictions


def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y:start_y + min_dim, start_x:start_x + min_dim]


def load_video(path, max_frames=0, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames) / 255.0


def load_video_new(path, lable):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == 0:
                break
    finally:
        cap.release()
    if len(frames) >= 80:
        outputframes = frames[0:80]
    else:
        beishu = math.floor(80 / len(frames))
        outputframes = []
        for cishu in range(0, beishu):
            outputframes.extend(frames)
        existlen = len(outputframes)
        outputframes.extend(frames[0:80 - existlen])
    return np.array(outputframes) / 255.0, lable


def load_video_1019(path):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, (224, 224))
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == 0:
                break
    finally:
        cap.release()
    if len(frames) >= 80:
        outputframes = frames[0:80]
    else:
        beishu = math.floor(80 / len(frames))
        outputframes = []
        for cishu in range(0, beishu):
            outputframes.extend(frames)
        existlen = len(outputframes)
        outputframes.extend(frames[0:80 - existlen])
    return np.array(outputframes) / 255.0


def get_batch_1005(video_list, lable_list, batch_size):
    video_tensor, lable_tensor = tf.cast(video_list, tf.string), tf.cast(lable_list, tf.int32)
    # print(video_tensor)
    input_queue = tf.train.slice_input_producer([video_tensor, lable_tensor], shuffle=False)
    video = input_queue[0]
    label = input_queue[1]

    video_batch, label_batch = tf.train.batch([video, label], batch_size=batch_size, capacity=20, num_threads=1)
    return video_batch, label_batch


def next_batch():
    videofile = open("./videofile.txt", "r+", encoding="utf8")
    video_list = [str(i.strip()) for i in videofile.readlines()]
    lablefile = open("./lablefile.txt", "r+", encoding="utf8")
    lable_list = [int(i.strip()) for i in lablefile.readlines()]

    # video_list = tf.constant(video_list)
    # lable_list = tf.constant(lable_list)

    dataset = tf.data.Dataset.from_tensor_slices((video_list, lable_list))
    # dataset = dataset.map(load_video_new)

    dataset = dataset.batch(5)

    # iterator = tf.contrib.eager(dataset)

    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def get_video_input(video_list):
    video_input = load_video_1019(video_list[0].decode())
    video_input = np.expand_dims(video_input, axis=0)
    for count in range(1, 5):
        one = load_video_1019(video_list[count].decode())
        video_input = np.insert(video_input, count, one, axis=0)
    return video_input


# loss计算
# 传入参数：logits，网络计算输出值。labels，真实值，0或者1
# 返回参数：loss，损失值
def losses(logits, labels):
    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                       name='xentropy_per_example')
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.compat.v1.summary.scalar(scope.name + '/loss', loss)
    return loss


# loss损失值优化
# 输入参数：loss。learning_rate，学习速率。
# 返回参数：train_op，训练op，这个参数要输入sess.run中让模型去训练。
def trainningold(loss, learning_rate):
    with tf.name_scope('optimizer'):
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # learning_rate = tf.train.exponential_decay(1e-2, global_step, decay_steps=sample_size / batch, decay_rate=0.9,staircase=True)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=30, decay_rate=0.9,
                                                   staircase=True)

        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


iter_count = 0


def trainning(loss, learning_rate, global_step):
    with tf.name_scope('optimizer'):
        # global_step = tf.Variable(global_step.item(), name='global_step', trainable=False)
        # learning_rate = tf.train.exponential_decay(1e-2, global_step, decay_steps=sample_size / batch, decay_rate=0.9,staircase=True)
        learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps=30, decay_rate=0.9,
                                                   staircase=True)
        global iter_count
        global_step = tf.Variable(iter_count, name='global_step', trainable=False)
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


# 评价/准确率计算
# 输入参数：logits，网络计算值。labels，标签，也就是真实值，在这里是0或者1。
# 返回参数：accuracy，当前step的平均准确率，也就是在这些batch中多少张图片被正确分类了。
def evaluation(logits, labels):
    with tf.variable_scope('accuracy') as scope:
        correct = tf.nn.in_top_k(logits, labels, 1)
        # correct = tf.nn.in_top_k(logits, labels, 5)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float16))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy


import time


# 写入日志
def write_log(logpath, content):
    timestr = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    file = open(logpath, "a+", encoding="utf8")
    file.write(content + "\t" + timestr + "\n")
    file.close()


if __name__ == "__main__":
    checkpoint_path = "/cdrom/lys/kineticsattengroup002br1/checkpoint1005"

    videofile = open("./videofile.txt", "r+", encoding="utf8")
    video_list = [str(i.strip()) for i in videofile.readlines()]
    lablefile = open("./lablefile.txt", "r+", encoding="utf8")
    lable_list = [int(i.strip()) for i in lablefile.readlines()]
    linesnum = len(lable_list)

    # video_list = tf.constant(video_list)
    # lable_list = tf.constant(lable_list)

    dataset = tf.data.Dataset.from_tensor_slices(tensors=(video_list, lable_list))
    # dataset = dataset.map(load_video_new)

    # dataset = dataset.shuffle(buffer_size=3)

    dataset = dataset.batch(5)
    dataset = dataset.repeat(100)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()

    video_ph = tf.placeholder(tf.float32, shape=(5, 80, 224, 224, 3))
    lable_ph = tf.placeholder(tf.int32, shape=(5))
    step_ph = tf.placeholder(tf.int32)

    rgb_model = InceptionI3d(num_classes=7, spatial_squeeze=True, final_endpoint='Logits')
    rgb_logits, _ = rgb_model(video_ph, is_training=True, dropout_keep_prob=0.5)
    train_loss = losses(rgb_logits, lable_ph)
    # train_op = trainning(train_loss, 0.5)
    train_op = trainning(train_loss, 1e-2, step_ph)
    train_acc = evaluation(rgb_logits, lable_ph)

    sess = tf.compat.v1.Session()
    # 产生一个saver来存储训练好的模型
    saver = tf.compat.v1.train.Saver(max_to_keep=1)

    sess.run(tf.compat.v1.global_variables_initializer())
    # # 开启协调器，用于管理多线程，在TF中tf.Coordinator和 tf.QueueRunner通常一起使用
    coord = tf.train.Coordinator()
    # # 一定要调用入队线程启动器tf.train.start_queue_runners启动队列填充！！！
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    checkpoint_path = os.path.join(checkpoint_path, 'i3d.ckpt')
    try:
        for i in range(100):
            for l in range(linesnum // 5):
                video_list, lable_list = sess.run(one_element)
                video_input = get_video_input(video_list)
                _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc],
                                                feed_dict={video_ph: video_input, lable_ph: lable_list, step_ph: i})
                # sess.run([train_op, train_loss, train_acc],feed_dict = {video_ph:video_input,lable_ph:lable_list})
                write_log("/cdrom/lys/kineticsattengroup002br1/1005.log",
                          'Step %d, batch_num %d, train loss = %.2f, train accuracy = %.2f%%' % (
                          i, l, tra_loss, tra_acc * 100.0))
            write_log("/cdrom/lys/kineticsattengroup002br1/1005.log",
                      'Step %d, train loss = %.2f, train accuracy = %.2f%%' % (i, tra_loss, tra_acc * 100.0))
            # print('Step %d, train loss = %.2f, train accuracy = %.2f%%' % (i, train_loss, train_acc * 100.0))
            # checkpoint_path = os.path.join(checkpoint_path, 'i3d.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)
            iter_count += 1

    except Exception as ex:
        print('Done Training')
        print(ex)
    finally:
        pass
        # 协调器coord发出所有线程终止信号
        coord.request_stop()
    # 把开启的线程加入主线程，等待threads结束
    coord.join(threads)
    # 关闭session
    sess.close()
