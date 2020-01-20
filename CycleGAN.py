from __future__ import absolute_import, division, print_function
import os
import glob
import cv2
import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Conv2D, Input, Layer, Dense, Flatten, Conv2DTranspose
from tensorflow.keras.models import Model,Sequential
import tensorflow_addons as tfa
import numpy as np
import random
import sys



class ProcessImage:
    def __init__(self):
        pass

    def load_image(self):
        dom_A = glob.glob("image_domainA/*")
        dom_B = glob.glob("image_domainB/*")
        image_A = [cv2.imread(i) for i in dom_A]
        image_B = [cv2.imread(i) for i in dom_B]
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
        self.norm=tfa.layers.InstanceNormalization(axis=3, 
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
        self.lamda = lamda
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
        loss_l1 = (loss_2 * self.lamda) - loss_1
        return loss_l1

class CycleGAN:
    def __init__(self):
        self.Gen_G = Generator(256, 3)
        self.Gen_F = Generator(256, 3)
        self.Dis_G = Discriminator(256, 3)
        self.Dis_F = Discriminator(256, 3)
        self.Loss_cycle_1 = Lossfunc(10)
        self.Loss_cycle_2 = Lossfunc(10)
        
        self.model_1 = Sequential([
            self.Gen_G.model,
            self.Dis_G.model
        ])
        self.model_2 = Sequential([
            self.Gen_F.model,
            self.Dis_F.model
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
        self.model_1.layers[1].trainable = True
        self.model_2.layers[1].trainable = True
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
                
        
    def train(self, input_A, input_B, batch_size=1, steps=1):
        self.Loss_cycle_1.set_input_data(input_A)
        self.Loss_cycle_2.set_input_data(input_B)
        self.Loss_cycle_1.set_batch_size(batch_size)
        self.Loss_cycle_2.set_batch_size(batch_size)
        b = np.array([0] * len(input_B), dtype=np.float32)
        for i in range(steps):
            print("Step : ", i + 1)
            label_disG, datas_DisG = self.make_randomdata(input_A, self.Gen_G.model, self.Gen_F.model)
            label_disF, datas_DisF = self.make_randomdata(input_B, self.Gen_F.model, self.Gen_G.model)
            self.Dis_G.model.fit(datas_DisG, label_disG, batch_size=2)
            self.Dis_F.model.fit(datas_DisF, label_disF, batch_size = 2)
            self.Loss_cycle_1.predict_data()
            self.Loss_cycle_2.predict_data()
            self.model_1.fit(input_A, b, batch_size=2)
            self.model_2.fit(input_B, b, batch_size=2)      


if __name__ == "__main__":
    processor = ProcessImage()
    A, B = processor.load_image()
    #tf.executing_eagerly()
    #tf.enable_eager_execution()
    cycleGAN = CycleGAN()
    cycleGAN.Loss_cycle_1.set_input_data(A[0][0:4])
    cycleGAN.Loss_cycle_2.set_input_data(B[0][0:4])
    #cycleGAN.Loss_cycle_1.set_predict_data(A[0][0])
    #cycleGAN.Loss_cycle_2.set_predict_data(B[0][0])
    cycleGAN.compile_model()
    cycleGAN.train(A[0][0:4], B[0][0:4], batch_size=2, steps=2)