from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import sys
import numpy as np
import glob


def loadimage():
    datagen = ImageDataGenerator(rotation_range=45,
                                 width_shift_range=0.15, height_shift_range=0.15, shear_range=0.1, zoom_range=0.2, horizontal_flip=False, fill_mode='nearest')

    for i in range(2):
        #trainlabel = [0 for i in range(6)]
        #trainlabel[i] = 1
        list_domain = ["A", "B"]
        file_t = glob.glob("image_domain" + str(list_domain[i]) + "/*")
        save_dir = "images_generated_" + str(list_domain[i])
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        for s in range(len(file_t)):
            traingenerator = datagen.flow(
                np.array([cv2.imread(file_t[s])]),
                batch_size=1
            )
            batches = traingenerator
            g_img = batches[0].astype(np.uint8)
            imagename = "images_" + str(i) + "_" + str(s) + ".png"
            output_dir = os.path.join(save_dir, imagename)
            cv2.imwrite(output_dir, g_img[0])


loadimage()