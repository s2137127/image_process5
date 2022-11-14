from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
import matplotlib as mpb
from mainui import Ui_MainWindow
import cv2
import numpy as np
import imageio.v2 as imageio
import time
from scipy.spatial import distance
import pyqtgraph as pg

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.img = None
        self.img_path = None
        self.xyz=None
        self.setupUi(self)
        self.pushButton_origin.clicked.connect(self.pushbuttom_origin_clicked)
        self.pushButton_RGB.clicked.connect(self.pushbuttom_RGB_clicked)
        self.pushButton_HSI.clicked.connect(self.pushbuttom_HSI_clicked)
        self.pushButton_CMY.clicked.connect(self.pushbuttom_CMY_clicked)
        self.pushButton_YUV.clicked.connect(self.pushbuttom_YUV_clicked)
        self.pushButton_XYZ.clicked.connect(self.pushbuttom_XYZ_clicked)
        self.pushButton_Lab.clicked.connect(self.pushbuttom_lab_clicked)
        self.pushButton_kmean.clicked.connect(self.pushbuttom_kmean_clicked)
        self.pushButton_prob2.clicked.connect(self.pushbuttom_prob2_clicked)
        self.comboBox_color.insertItem(0, 'RdYlBu')#set color option in comboBox
        self.comboBox_color.insertItem(1,'PiYG')
        self.comboBox_color.insertItem(2, 'PRGn')
        self.comboBox_color.insertItem(3, 'BrBG')
        self.comboBox_color.insertItem(4, 'bwr')
    def pushbuttom_origin_clicked(self):
        self.img_path, _ = QFileDialog.getOpenFileName(self,
                                                       "Open file",
                                                       "./",
                                                       "Images (*.png *.BMP *.jpg *.JPG)")

        img = imageio.imread(self.img_path)  # 讀檔
        self.img = np.array(img).astype(np.float64)
        height, width, channel = self.img.shape
        self.gray = np.zeros((self.img.shape[0], self.img.shape[1]), np.uint8)
        for i in range(self.gray.shape[0]):
            for j in range(self.gray.shape[1]):
                self.gray[i, j] = np.mean(self.img[i, j, :])
        qimg = QImage(img.data, width, height, 3 * width, QImage.Format_RGB888)
        self.label_origin.setPixmap(QPixmap.fromImage(qimg))
        self.label_origin.setScaledContents(True)


    def pushbuttom_RGB_clicked(self):#show rgb img
        height, width, channel = self.img.shape
        out = self.img.copy().astype(np.uint8)
        qimg = QImage(out, width, height, 3 * width, QImage.Format_RGB888)
        self.label_prob1.setPixmap(QPixmap.fromImage(qimg))
        self.label_prob1.setScaledContents(True)

    def pushbuttom_HSI_clicked(self):#show hsi img
        height, width, channel = self.img.shape
        hsi = np.zeros_like(self.img)

        for i in range(self.img.shape[0]):#get hsi
            for j in range(self.img.shape[1]):
                R, G, B = self.img[i, j, 0], self.img[i, j, 1], self.img[i, j, 2]
                # print(type(G))
                # print(type(R))
                I = np.mean(self.img[i, j, :])
                S = 0
                if I > 0:
                    S = 1 - np.min(self.img[i, j, :]) / I
                    S *= 255
                H = np.arccos((R - 0.5 * G - 0.5 * B) / (
                        R ** 2 + G ** 2 + B ** 2 - R * G - R * B - G * B + 0.00001) ** 0.5) * 180 / np.pi
                # print(H)
                if B > G:
                    H = 360 - H
                H *= 255 / 360

                hsi[i, j, 0] = H
                # print(hsi[i, j, 0])

                hsi[i, j, 1] = S
                # print(hsi[i, j, 1])
                hsi[i, j, 2] = I
        out = hsi.astype(np.uint8)
        qimg = QImage(out.data, width, height, 3 * width, QImage.Format_RGB888)
        self.label_prob1.setPixmap(QPixmap.fromImage(qimg))
        self.label_prob1.setScaledContents(True)

    def pushbuttom_CMY_clicked(self):#show cmy img
        height, width, channel = self.img.shape
        cmy = np.zeros_like(self.img)

        for i in range(self.img.shape[0]):#get cmy
            for j in range(self.img.shape[1]):
                R, G, B = self.img[i, j, 0] / 255, self.img[i, j, 1] / 255, self.img[i, j, 2] / 255

                C = 1 - R
                M = 1 - G
                Y = 1 - B
                cmy[i, j, :] = [C, M, Y]
        cmy *= 255
        out = cmy.astype(np.uint8)
        qimg = QImage(out.data, width, height, 3 * width, QImage.Format_RGB888)
        self.label_prob1.setPixmap(QPixmap.fromImage(qimg))
        self.label_prob1.setScaledContents(True)

    def pushbuttom_YUV_clicked(self):#show yuv img
        height, width, channel = self.img.shape
        yuv = np.zeros_like(self.img)
        for i in range(self.img.shape[0]):#get yuv
            for j in range(self.img.shape[1]):
                R, G, B = self.img[i, j, 0], self.img[i, j, 1], self.img[i, j, 2]
                Y = 0.257 * R + 0.504 * G + 0.098 * B + 16
                U = -0.148 * R - 0.291 * G + 0.439 * B + 128
                V = 0.439 * R - 0.368 * G - 0.071 * B + 128
                yuv[i, j, :] = [Y, U, V]
        out = yuv.astype(np.uint8)
        qimg = QImage(out.data, width, height, 3 * width, QImage.Format_RGB888)
        self.label_prob1.setPixmap(QPixmap.fromImage(qimg))
        self.label_prob1.setScaledContents(True)


    def pushbuttom_XYZ_clicked(self):#show xyz img
        height, width, channel = self.img.shape
        self.xyz = np.zeros_like(self.img)
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):#get xyz
                R, G, B = self.img[i, j, 0], self.img[i, j, 1], self.img[i, j, 2]
                X = 0.412453 * R + 0.357580 * G + 0.180423 * B
                Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
                Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
                self.xyz[i, j, :] = [X, Y, Z]
        out = self.xyz.copy().astype(np.uint8)
        qimg = QImage(out.data, width, height, 3 * width, QImage.Format_RGB888)
        self.label_prob1.setPixmap(QPixmap.fromImage(qimg))
        self.label_prob1.setScaledContents(True)

    def pushbuttom_lab_clicked(self):#show Lab img
        height, width, channel = self.img.shape
        if self.xyz is None:#get xyz if xyz haven't get
            self.xyz = np.zeros_like(self.img)
            for i in range(self.img.shape[0]):
                for j in range(self.img.shape[1]):
                    R, G, B = self.img[i, j, 0], self.img[i, j, 1], self.img[i, j, 2]
                    X = 0.412453 * R + 0.357580 * G + 0.180423 * B
                    Y = 0.212671 * R + 0.715160 * G + 0.072169 * B
                    Z = 0.019334 * R + 0.119193 * G + 0.950227 * B
                    self.xyz[i, j, :] = [X, Y, Z]
        # print(self.xyz)
        xyz = self.xyz / 255.0
        lab = np.zeros_like(xyz).astype(float)
        for i in range(lab.shape[0]):#get lab
            for j in range(lab.shape[1]):
                x, y, z = xyz[i, j, 0], xyz[i, j, 1], xyz[i, j, 2]
                x /= 0.95047
                z /= 1.08883
                L = 116 * self.h(y) - 16
                a = 500 * (self.h(x) - self.h(y))
                b = 200 * (self.h(y) - self.h(z))
                lab[i,j,:] = [L,a,b]
        out = lab.astype(np.uint8)
        qimg = QImage(out.data, width, height, 3 * width, QImage.Format_RGB888)
        self.label_prob1.setPixmap(QPixmap.fromImage(qimg))
        self.label_prob1.setScaledContents(True)

    def h(self, x):
        a = x**(1/3) if x > 0.008856 else 7.787 * x + 16 / 116
        return a

    def pushbuttom_prob2_clicked(self):
        level = self.spinBox_level.value()
        gray = self.gray.copy()*(level/256)
        img = np.zeros_like(self.img)
        color = self.comboBox_color.currentText()
        colorbar = np.zeros((level,10,3))
        # print("1")
        cmap = plt.get_cmap(color)
        norm = plt.Normalize(0, level)
        # print("1")
        for i in range(level):#make colorbar
            c = cmap(norm(i))[:-1]
            colorbar[i,:,:] =  [c]*10
        colorbar *= 255
        for i in range(gray.shape[0]):#make pseudo color image
            for j in range(gray.shape[1]):
                img[i,j] = cmap(norm(int(gray[i,j])))[:-1]
        img *= 255
        img = img.astype(np.uint8)
        colorbar = colorbar.astype(np.uint8)
        qimg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
        self.label_color.setPixmap(QPixmap.fromImage(qimg))
        self.label_color.setScaledContents(True)
        qimg = QImage(self.gray.data, gray.shape[1], gray.shape[0], QImage.Format_Grayscale8)
        self.label_gray.setPixmap(QPixmap.fromImage(qimg))
        self.label_gray.setScaledContents(True)
        qimg = QImage(colorbar.data, colorbar.shape[1], colorbar.shape[0], 3 * colorbar.shape[1], QImage.Format_RGB888)
        self.label_colorbar.setPixmap(QPixmap.fromImage(qimg))
        self.label_colorbar.setScaledContents(True)
        self.lineEdit_2.setText(str(int(level/2)))
        self.lineEdit_3.setText(str(level))

    def pushbuttom_kmean_clicked(self):
        k=self.spinBox_k.value()
        # print(k)
        color = np.random.randint(0, 255, size=(k, 3))
        mean = self.kmeans(k)
        img = np.zeros_like(self.img)
        for i in range(img.shape[0]):#get every pixel of output img  by kmean
            for j in range(img.shape[1]):
                dst = [distance.euclidean(mean[idx], self.img[i, j]) for idx in range(k)]
                img[i,j]=color[np.argmin(dst)]
                # print(img[i,j])
        # print(img)
        img = img.astype(np.uint8)
        # print(img)
        qimg = QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QImage.Format_RGB888)
        self.label_kmean.setPixmap(QPixmap.fromImage(qimg))
        self.label_kmean.setScaledContents(True)

    def kmeans(self,k):#kmean algorithm
        img = self.img.copy()
        r_x = np.random.randint(0, img.shape[0], size=k)#get init mean vector
        r_y = np.random.randint(0, img.shape[1], size=k)
        mean_b=[img[r_x[i], r_y[i], :] for i in range(k)]
        label = {i:[img[r_x[i], r_y[i], :]] for i in range(k)}

        for i in range(img.shape[0]):#classify each pixel to k label
            for j in range(img.shape[1]):
                dst = [distance.euclidean(mean_b[i], img[i, j]) for i in range(k)]
                label[int(np.argmin(dst))].append(img[i, j, :])
        mean_a = [np.mean(label[i], axis=0) for i in range(k)]#update mean point
        while np.mean(abs(np.array(mean_b) - np.array(mean_a))) > 5:#while dst(before mean,after mean)>5
            mean_b = mean_a.copy()
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    dst = [distance.euclidean(mean_b[idx], self.img[i, j]) for idx in range(k)]
                    label[int(np.argmin(dst))].append(img[i, j, :])
            mean_a = [np.mean(label[i], axis=0) for i in range(k)]#update mean point
        #return k mean points
        return mean_a

if __name__ == '__main__':
    app = QApplication([])
    w = MainWindow()
    w.show()
    app.exec()
