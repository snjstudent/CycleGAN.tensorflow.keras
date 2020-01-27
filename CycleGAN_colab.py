!pip install -U -q PyDrive

#!pip install --upgrade google-api-python-client

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

import os
import shutil


# Colaboratory側に格納用ディレクトリを事前に作成しておく
local_folder_name = 'image_domainA'
os.mkdir(local_folder_name)

# tmpにあるdf_sample.csvとgoogle_banner.pngをDownloadする
# 親フォルダーのtmpのIDを取得する
tmp_folder_name = 'image_domainA'
tmp_id = drive.ListFile({'q': "'root' in parents and title= '{folder_id}' and trashed=False  and mimeType='application/vnd.google-apps.folder' ".format(folder_id=tmp_folder_name)}).GetList()[0]['id']

# GoogleDrive上のtmpフォルダー内にあるファイルの一覧を取得してくる
list_files = drive.ListFile({'q': "'{folder_id}' in parents and trashed=False  and mimeType !='application/vnd.google-apps.folder' ".format(folder_id=tmp_id)}).GetList()

# ローカルのrootにダウンロードしてきて、指定したファイルに移動させる
for file_obj in list_files:
  _title = file_obj['title']
  _id = file_obj['id']
  _download = drive.CreateFile({'id': _id})
  _download.GetContentFile(_title)
  # GoogleDriveのファイル名をそのまま移動する
  shutil.move(_title, local_folder_name)
 
 
 # Colaboratory側に格納用ディレクトリを事前に作成しておく
local_folder_name = 'image_domainB'
os.mkdir(local_folder_name)

# tmpにあるdf_sample.csvとgoogle_banner.pngをDownloadする
# 親フォルダーのtmpのIDを取得する
tmp_folder_name = 'image_domainB'
tmp_id = drive.ListFile({'q': "'root' in parents and title= '{folder_id}' and trashed=False  and mimeType='application/vnd.google-apps.folder' ".format(folder_id=tmp_folder_name)}).GetList()[0]['id']

# GoogleDrive上のtmpフォルダー内にあるファイルの一覧を取得してくる
list_files = drive.ListFile({'q': "'{folder_id}' in parents and trashed=False  and mimeType !='application/vnd.google-apps.folder' ".format(folder_id=tmp_id)}).GetList()

# ローカルのrootにダウンロードしてきて、指定したファイルに移動させる
for file_obj in list_files:
  _title = file_obj['title']
  _id = file_obj['id']
  _download = drive.CreateFile({'id': _id})
  _download.GetContentFile(_title)
  # GoogleDriveのファイル名をそのまま移動する
  shutil.move(_title, local_folder_name)

from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model,Sequential
#import tensorflow_addons as tfa
import numpy as np
import random
import sys

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Orginal implementation from keras_contrib/layer/normalization
# =============================================================================

import logging
import tensorflow as tf



class GroupNormalization(tf.keras.layers.Layer):
    """Group normalization layer.
    Group Normalization divides the channels into groups and computes
    within each group the mean and variance for normalization.
    Empirically, its accuracy is more stable than batch norm in a wide
    range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Relation to Layer Normalization:
    If the number of groups is set to 1, then this operation becomes identical
    to Layer Normalization.
    Relation to Instance Normalization:
    If the number of groups is set to the
    input dimension (number of groups is equal
    to number of channels), then this operation becomes
    identical to Instance Normalization.
    Arguments
        groups: Integer, the number of groups for Group Normalization.
            Can be in the range [1, N] where N is the input dimension.
            The input dimension must be divisible by the number of groups.
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
    References
        - [Group Normalization](https://arxiv.org/abs/1803.08494)
    """

    def __init__(self,
                 groups=2,
                 axis=-1,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.groups = groups
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = tf.keras.initializers.get(beta_initializer)
        self.gamma_initializer = tf.keras.initializers.get(gamma_initializer)
        self.beta_regularizer = tf.keras.regularizers.get(beta_regularizer)
        self.gamma_regularizer = tf.keras.regularizers.get(gamma_regularizer)
        self.beta_constraint = tf.keras.constraints.get(beta_constraint)
        self.gamma_constraint = tf.keras.constraints.get(gamma_constraint)
        self._check_axis()

    def build(self, input_shape):

        self._check_if_input_shape_is_none(input_shape)
        self._set_number_of_groups_for_instance_norm(input_shape)
        self._check_size_of_dimensions(input_shape)
        self._create_input_spec(input_shape)

        self._add_gamma_weight(input_shape)
        self._add_beta_weight(input_shape)
        self.built = True
        super().build(input_shape)

    def call(self, inputs):

        input_shape = tf.keras.backend.int_shape(inputs)
        tensor_input_shape = tf.shape(inputs)

        reshaped_inputs, group_shape = self._reshape_into_groups(
            inputs, input_shape, tensor_input_shape)

        normalized_inputs = self._apply_normalization(reshaped_inputs,
                                                      input_shape)

        outputs = tf.reshape(normalized_inputs, tensor_input_shape)

        return outputs

    def get_config(self):
        config = {
            'groups':
            self.groups,
            'axis':
            self.axis,
            'epsilon':
            self.epsilon,
            'center':
            self.center,
            'scale':
            self.scale,
            'beta_initializer':
            tf.keras.initializers.serialize(self.beta_initializer),
            'gamma_initializer':
            tf.keras.initializers.serialize(self.gamma_initializer),
            'beta_regularizer':
            tf.keras.regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer':
            tf.keras.regularizers.serialize(self.gamma_regularizer),
            'beta_constraint':
            tf.keras.constraints.serialize(self.beta_constraint),
            'gamma_constraint':
            tf.keras.constraints.serialize(self.gamma_constraint)
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def compute_output_shape(self, input_shape):
        return input_shape

    def _reshape_into_groups(self, inputs, input_shape, tensor_input_shape):

        group_shape = [tensor_input_shape[i] for i in range(len(input_shape))]
        group_shape[self.axis] = input_shape[self.axis] // self.groups
        group_shape.insert(self.axis, self.groups)
        group_shape = tf.stack(group_shape)
        reshaped_inputs = tf.reshape(inputs, group_shape)
        return reshaped_inputs, group_shape

    def _apply_normalization(self, reshaped_inputs, input_shape):

        group_shape = tf.keras.backend.int_shape(reshaped_inputs)
        group_reduction_axes = list(range(1, len(group_shape)))
        axis = -2 if self.axis == -1 else self.axis - 1
        group_reduction_axes.pop(axis)

        mean, variance = tf.nn.moments(
            reshaped_inputs, group_reduction_axes, keepdims=True)

        gamma, beta = self._get_reshaped_weights(input_shape)
        normalized_inputs = tf.nn.batch_normalization(
            reshaped_inputs,
            mean=mean,
            variance=variance,
            scale=gamma,
            offset=beta,
            variance_epsilon=self.epsilon)
        return normalized_inputs

    def _get_reshaped_weights(self, input_shape):
        broadcast_shape = self._create_broadcast_shape(input_shape)
        gamma = None
        beta = None
        if self.scale:
            gamma = tf.reshape(self.gamma, broadcast_shape)

        if self.center:
            beta = tf.reshape(self.beta, broadcast_shape)
        return gamma, beta

    def _check_if_input_shape_is_none(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                             'input tensor should have a defined dimension '
                             'but the layer received an input with shape ' +
                             str(input_shape) + '.')

    def _set_number_of_groups_for_instance_norm(self, input_shape):
        dim = input_shape[self.axis]

        if self.groups == -1:
            self.groups = dim

    def _check_size_of_dimensions(self, input_shape):

        dim = input_shape[self.axis]
        if dim < self.groups:
            raise ValueError(
                'Number of groups (' + str(self.groups) + ') cannot be '
                'more than the number of channels (' + str(dim) + ').')

        if dim % self.groups != 0:
            raise ValueError(
                'Number of groups (' + str(self.groups) + ') must be a '
                'multiple of the number of channels (' + str(dim) + ').')

    def _check_axis(self):

        if self.axis == 0:
            raise ValueError(
                "You are trying to normalize your batch axis. Do you want to "
                "use tf.layer.batch_normalization instead")

    def _create_input_spec(self, input_shape):

        dim = input_shape[self.axis]
        self.input_spec = tf.keras.layers.InputSpec(
            ndim=len(input_shape), axes={self.axis: dim})

    def _add_gamma_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(
                shape=shape,
                name='gamma',
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint)
        else:
            self.gamma = None

    def _add_beta_weight(self, input_shape):

        dim = input_shape[self.axis]
        shape = (dim,)

        if self.center:
            self.beta = self.add_weight(
                shape=shape,
                name='beta',
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint)
        else:
            self.beta = None

    def _create_broadcast_shape(self, input_shape):
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis] // self.groups
        broadcast_shape.insert(self.axis, self.groups)
        return broadcast_shape

class InstanceNormalization(GroupNormalization):
    """Instance normalization layer.
    Instance Normalization is an specific case of ```GroupNormalization```since
    it normalizes all features of one channel. The Groupsize is equal to the
    channel size. Empirically, its accuracy is more stable than batch norm in a
    wide range of small batch sizes, if learning rate is adjusted linearly
    with batch sizes.
    Arguments
        axis: Integer, the axis that should be normalized.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    Output shape
        Same shape as input.
    References
        - [Instance Normalization: The Missing Ingredient for Fast Stylization]
        (https://arxiv.org/abs/1607.08022)
    """

    def __init__(self, **kwargs):
        if "groups" in kwargs:
            logging.warning("The given value for groups will be overwritten.")

        kwargs["groups"] = -1
        super().__init__(**kwargs)




class ProcessImage:
    def __init__(self):
        pass

    def load_image(self):
        dom_A = glob.glob("image_domainA/*")
        dom_B = glob.glob("image_domainB/*")
        image_A = [cv2.imread(i) for i in dom_A]
        image_B = [cv2.imread(i) for i in dom_B]
        image_A = image_B[0:len(image_B)]
        image_A = [cv2.resize(i,(256, 256))/255 for i in image_A]
        image_B = [cv2.resize(i,(256,256))/255 for i in image_B]
        return np.array([image_A],dtype=np.float32), np.array([image_B],dtype=np.float32)

class ConvBlock(Model):
    def __init__(self, filter_num: int, kernel_size: int = 3, stride: int = 1,do_frac=False, do_leakyr=False ,*args, **kwargs):
        super(ConvBlock, self).__init__(*args, **kwargs)
        self.filter_num = filter_num
        self.kernel_size = kernel_size
        self.stride = stride
        self.do_frac = do_frac
        self.do_leakyr = do_leakyr
        if self.do_frac:
            self.conv = Conv2DTranspose(filters=self.filter_num, kernel_size=self.kernel_size, strides=int(self.stride ** (-1)),padding='same')
        else:
            self.conv = Conv2D(filters=self.filter_num, kernel_size=self.kernel_size, strides=self.stride,padding='same')
        self.norm=InstanceNormalization(axis=3, 
                                               center=True, 
                                               scale=True,
                                               beta_initializer="random_uniform",
                                               gamma_initializer="random_uniform")
        if self.do_leakyr:
            self.dense = Dense(filter_num, activation=tf.nn.leaky_relu)
        else:
            self.dense = Dense(filter_num, activation=tf.nn.relu)
    
    def call(self, inputs):
        tensor = self.conv(inputs)
        tensor = self.norm(tensor)
        return self.dense(tensor)


class ResNN(Model):
    def __init__(self, unit_num=9, *args, **kwargs):
        super(ResNN, self).__init__(*args, **kwargs)
        self.unit_num = unit_num
        self.conv_res = Conv2D(filters=256, kernel_size=(3, 3),padding='same')
        self.resList_1: List[Layers] = []
        self.resList_2: List[Layers] = []
    
    def call(self, inputs):
        for i in range(self.unit_num // 2):
            self.resList_1.append(self.conv_res(inputs))
            inputs = self.resList_1[-1]
        inputs = self.conv_res(inputs)
        for i in range(self.unit_num // 2):
            inputs += self.resList_1[-(i + 1)]
            self.resList_2.append(self.conv_res(inputs))
            inputs = self.resList_2[-1]
        return inputs
        
    
class Generator:
    def __init__(self, img_dim, img_channel):
        self.img_dim = img_dim
        self.img_channel = img_channel
        conv_block_1 = ConvBlock(64, 7)
        conv_block_2 = ConvBlock(128, stride=2)
        conv_block_3 = ConvBlock(256, stride=2)
        conv_block_4 = ConvBlock(128, stride=1 / 2, do_frac=True)
        conv_block_5 = ConvBlock(64, stride=1 / 2, do_frac=True)
        conv_block_6 = ConvBlock(3, 7)
        resnet = ResNN(unit_num=9)

        self.input_g = Input(shape=(img_dim, img_dim, img_channel), name="input_generator")
        conv_1 = conv_block_1(self.input_g)
        conv_2 = conv_block_2(conv_1)
        conv_3 = conv_block_3(conv_2)
        res_1 = resnet(conv_3)
        conv_4 = conv_block_4(res_1)
        conv_5 = conv_block_5(conv_4)
        self.conv_6 = conv_block_6(conv_5)
        self.model = Model(inputs=[self.input_g], outputs=[self.conv_6])
        #self.model.summary()
        



class Discriminator:
    def __init__(self, img_dim, img_channel):
        self.img_dim = img_dim
        self.img_channel = img_channel
        list_conv: List[Model] = []
        input_d = Input(shape=(img_dim, img_dim, img_channel))
        conv_1 = ConvBlock(64, 4,do_leakyr=True)
        list_conv.append(conv_1(input_d))
        for i in range(3):
            conv_tmp = ConvBlock(64 * (2 ** (i + 1)), 4, do_leakyr=True)
            list_conv.append(conv_tmp(list_conv[-1]))
        flat = Flatten()(list_conv[-1])
        self.out = Dense(1, activation='sigmoid')(flat)
        self.model = Model(inputs=[input_d], outputs=[self.out])
        #self.model.summary()

class Lossfunc:
    def __init__(self, lamda):
        self.lamda = 10
        self.count = 1
        self.batch_size = 1

    def get_target_model(self, model_target):
        self.model_target = model_target
    
    def get_self_model(self, self_model):
        self.self_model = self_model
        
    def set_input_data(self, input_data):
        self.input_data = input_data
        self.pred_datas = input_data
    
    def set_predict_data(self, input_data):
        self.pred_data = input_data
    
    def set_batch_size(self, batch_size):
        self.count = 1
        self.batch_size = batch_size

    def predict_data(self):
        self.pred_datas = self.model_target.predict(self.self_model.predict(self.input_data))
    
    def l1_norm_loss(self, y_true, y_pred):
        loss_1 = tfk.losses.binary_crossentropy(y_true, y_pred)
        loss_2 = tfk.losses.mae(self.input_data[((self.count - 1) * self.batch_size):(self.count * self.batch_size)], self.pred_datas[((self.count - 1) * self.batch_size):(self.count * self.batch_size)])
        loss_2 = tf.reduce_mean(loss_2, axis=[1,2])
        self.count += 1
        if (self.count * self.batch_size >= len(self.input_data)):
            self.count = 1
        loss_l1 = (loss_2 * self.lamda) - loss_1
        return loss_l1

class CycleGAN:
    def __init__(self):
        self.Gen_G = Generator(256, 3)
        self.Gen_F = Generator(256, 3)
        self.Dis_G = Discriminator(256, 3)
        self.Dis_F = Discriminator(256, 3)
        self.Dis_G_1 = Discriminator(256, 3)
        self.Dis_F_1 = Discriminator(256, 3)
        self.Loss_cycle_1 = Lossfunc(10)
        self.Loss_cycle_2 = Lossfunc(10)
        
        self.model_1 = Sequential([
            self.Gen_G.model,
            self.Dis_G_1.model
        ])
        self.model_2 = Sequential([
            self.Gen_F.model,
            self.Dis_F_1.model
        ])
    def compile_model(self):
        #self.mode_tm = Model(inputs=[Gen_G.input_g, Gen_G.conv_6], outputs=[Gen_G.conv_6,Dis_G.out])
        self.Gen_G.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
        self.Gen_F.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
        self.Loss_cycle_1.get_target_model(self.Gen_F.model)
        self.Loss_cycle_1.get_self_model(self.Gen_G.model)
        self.Loss_cycle_2.get_target_model(self.Gen_G.model)
        self.Loss_cycle_2.get_self_model(self.Gen_F.model)
        self.model_1.layers[1].trainable = False
        self.model_2.layers[1].trainable = False
        self.model_1.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss=self.Loss_cycle_1.l1_norm_loss)
        self.model_2.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss=self.Loss_cycle_2.l1_norm_loss)
        self.Dis_G.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
        self.Dis_F.model.compile(
        optimizer=tfk.optimizers.Adam(lr=0.0001),
        loss='binary_crossentropy')
    
    def make_randomdata(self, input_, model_self, model_target):
        pred_datas = model_target.predict(model_target.predict(input_))
        datas = []
        label = []
        for i in range(len(input_)):
            r = random.uniform(0, 1)
            if r > 0.5:
                datas.append(pred_datas[i])
                label.append(0)
            else:
                datas.append(input_[i])
                label.append(1)
        return np.array(label, dtype=np.float32), np.array(datas, dtype=np.float32)
                
    def save_model(self,i):
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        drive = GoogleDrive(gauth)
        upload_file = drive.CreateFile()
        self.Dis_G.model.save_weights("Dis_G_"+str(i)+".hdf5")
        self.Dis_F.model.save_weights("Dis_F_"+str(i)+".hdf5")
        self.model_1.save_weights("model_1_"+str(i)+".hdf5")
        self.model_2.save_weights("model_2_"+str(i)+".hdf5")
        upload_file.SetContentFile("Dis_G_"+str(i)+".hdf5")
        upload_file.Upload()
        upload_file = drive.CreateFile()
        upload_file.SetContentFile("Dis_F_"+str(i)+".hdf5")
        upload_file.Upload()
        upload_file = drive.CreateFile()
        upload_file.SetContentFile("model_1_"+str(i)+".hdf5")
        upload_file.Upload()
        upload_file = drive.CreateFile()
        upload_file.SetContentFile("model_2_"+str(i)+".hdf5")
        upload_file.Upload()

    def train(self, input_A, input_B,count, batch_size=1, steps=1):
        self.Loss_cycle_1.set_input_data(input_A)
        self.Loss_cycle_2.set_input_data(input_B)
        self.Loss_cycle_1.set_batch_size(batch_size)
        self.Loss_cycle_2.set_batch_size(batch_size)
        b = np.array([0] * len(input_B), dtype=np.float32)
        self.Loss_cycle_1.batch_size = batch_size
        self.Loss_cycle_2.batch_size = batch_size
        for i in range(steps):
            #print("Step : ", i + 1)
            label_disG, datas_DisG = self.make_randomdata(input_A, self.Gen_G.model, self.Gen_F.model)
            if count%3==0:
              label_disF, datas_DisF = self.make_randomdata(input_B, self.Gen_F.model, self.Gen_G.model)
              self.Dis_G.model.fit(datas_DisG, label_disG, batch_size=batch_size)
              self.Dis_F.model.fit(datas_DisF, label_disF, batch_size=batch_size)
            self.model_1.layers[1].set_weights(self.Dis_G.model.get_weights())
            self.model_2.layers[1].set_weights(self.Dis_F.model.get_weights())
            self.Loss_cycle_1.predict_data()
            self.Loss_cycle_2.predict_data()
            self.model_1.fit(input_A, b, batch_size=batch_size)
            self.model_2.fit(input_B, b, batch_size=batch_size)
            testimage = self.Gen_F.model.predict(datas_DisG, batch_size=batch_size)
            cv2.imwrite("test.png", np.array(testimage[0]) * 255.0)
            print("")
            print("")
            print("image created")
            print("")
            print("")      


if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    processor = ProcessImage()
    A, B = processor.load_image()
    #tf.executing_eagerly()
    #tf.enable_eager_execution()
    cycleGAN = CycleGAN()
    cycleGAN.Loss_cycle_1.set_input_data(A[0][0:100])
    cycleGAN.Loss_cycle_2.set_input_data(B[0][0:100])
    #cycleGAN.Loss_cycle_1.set_predict_data(A[0][0])
    #cycleGAN.Loss_cycle_2.set_predict_data(B[0][0])
    cycleGAN.compile_model()
    for i in range(1,10000):
        
        if (i==1 or i%5==0):
          cycleGAN.save_model(i)

        print("Step : ", i + 1)
        A_batch = np.random.permutation(A[0])
        B_batch = np.random.permutation(B[0])
        for u in range(int(len(A_batch) / 100)):
            cycleGAN.Loss_cycle_1.set_input_data(A_batch[u * 100:(u + 1) * 100])
            cycleGAN.Loss_cycle_2.set_input_data(B_batch[u * 100:(u + 1) * 100])
            cycleGAN.train(A_batch[u * 100:(u + 1) * 100], B_batch[u * 100:(u + 1) * 100],u, batch_size=4, steps=1)