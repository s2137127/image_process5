from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib import pyplot as plt
import numpy as np
import imageio.v2 as imageio
from scipy.spatial import distance


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(993, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.pushButton_prob2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_prob2.setGeometry(QtCore.QRect(480, 500, 75, 23))
        self.pushButton_prob2.setObjectName("pushButton_prob2")
        self.layoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget.setGeometry(QtCore.QRect(60, 60, 201, 221))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_origin = QtWidgets.QLabel(self.layoutWidget)
        self.label_origin.setObjectName("label_origin")
        self.verticalLayout.addWidget(self.label_origin)
        self.pushButton_origin = QtWidgets.QPushButton(self.layoutWidget)
        self.pushButton_origin.setObjectName("pushButton_origin")
        self.verticalLayout.addWidget(self.pushButton_origin)
        self.layoutWidget1 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget1.setGeometry(QtCore.QRect(280, 60, 241, 261))
        self.layoutWidget1.setObjectName("layoutWidget1")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.layoutWidget1)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_prob1 = QtWidgets.QLabel(self.layoutWidget1)
        self.label_prob1.setObjectName("label_prob1")
        self.verticalLayout_2.addWidget(self.label_prob1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_RGB = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_RGB.setObjectName("pushButton_RGB")
        self.horizontalLayout.addWidget(self.pushButton_RGB)
        self.pushButton_HSI = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_HSI.setObjectName("pushButton_HSI")
        self.horizontalLayout.addWidget(self.pushButton_HSI)
        self.pushButton_CMY = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_CMY.setObjectName("pushButton_CMY")
        self.horizontalLayout.addWidget(self.pushButton_CMY)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_XYZ = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_XYZ.setObjectName("pushButton_XYZ")
        self.horizontalLayout_2.addWidget(self.pushButton_XYZ)
        self.pushButton_Lab = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_Lab.setObjectName("pushButton_Lab")
        self.horizontalLayout_2.addWidget(self.pushButton_Lab)
        self.pushButton_YUV = QtWidgets.QPushButton(self.layoutWidget1)
        self.pushButton_YUV.setObjectName("pushButton_YUV")
        self.horizontalLayout_2.addWidget(self.pushButton_YUV)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.verticalLayout_2.setStretch(0, 10)
        self.verticalLayout_2.setStretch(1, 1)
        self.verticalLayout_2.setStretch(2, 1)
        self.layoutWidget2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget2.setGeometry(QtCore.QRect(70, 330, 411, 191))
        self.layoutWidget2.setObjectName("layoutWidget2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.layoutWidget2)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label_gray = QtWidgets.QLabel(self.layoutWidget2)
        self.label_gray.setObjectName("label_gray")
        self.horizontalLayout_3.addWidget(self.label_gray)
        self.label_color = QtWidgets.QLabel(self.layoutWidget2)
        self.label_color.setObjectName("label_color")
        self.horizontalLayout_3.addWidget(self.label_color)
        self.label_colorbar = QtWidgets.QLabel(self.layoutWidget2)
        self.label_colorbar.setObjectName("label_colorbar")
        self.horizontalLayout_3.addWidget(self.label_colorbar)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.lineEdit = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit.setReadOnly(True)
        self.lineEdit.setObjectName("lineEdit")
        self.verticalLayout_4.addWidget(self.lineEdit)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_2.setReadOnly(True)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.verticalLayout_4.addWidget(self.lineEdit_2)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_4.addItem(spacerItem1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.layoutWidget2)
        self.lineEdit_3.setReadOnly(True)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.verticalLayout_4.addWidget(self.lineEdit_3)
        self.horizontalLayout_3.addLayout(self.verticalLayout_4)
        self.horizontalLayout_3.setStretch(0, 10)
        self.horizontalLayout_3.setStretch(1, 10)
        self.horizontalLayout_3.setStretch(2, 2)
        self.horizontalLayout_3.setStretch(3, 1)
        self.layoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.layoutWidget_2.setGeometry(QtCore.QRect(550, 90, 231, 231))
        self.layoutWidget_2.setObjectName("layoutWidget_2")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget_2)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_kmean = QtWidgets.QLabel(self.layoutWidget_2)
        self.label_kmean.setObjectName("label_kmean")
        self.verticalLayout_3.addWidget(self.label_kmean)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.spinBox_k = QtWidgets.QSpinBox(self.layoutWidget_2)
        self.spinBox_k.setProperty("value", 3)
        self.spinBox_k.setObjectName("spinBox_k")
        self.horizontalLayout_4.addWidget(self.spinBox_k)
        self.pushButton_kmean = QtWidgets.QPushButton(self.layoutWidget_2)
        self.pushButton_kmean.setObjectName("pushButton_kmean")
        self.horizontalLayout_4.addWidget(self.pushButton_kmean)
        self.horizontalLayout_4.setStretch(0, 2)
        self.horizontalLayout_4.setStretch(1, 4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.verticalLayout_3.setStretch(0, 9)
        self.verticalLayout_3.setStretch(1, 1)
        self.spinBox_level = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_level.setGeometry(QtCore.QRect(480, 480, 71, 21))
        self.spinBox_level.setSuffix("")
        self.spinBox_level.setMaximum(256)
        self.spinBox_level.setProperty("value", 256)
        self.spinBox_level.setObjectName("spinBox_level")
        self.comboBox_color = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_color.setGeometry(QtCore.QRect(480, 450, 69, 22))
        self.comboBox_color.setObjectName("comboBox_color")
        self.lineEdit_4 = QtWidgets.QLineEdit(self.centralwidget)
        self.lineEdit_4.setGeometry(QtCore.QRect(480, 430, 61, 20))
        self.lineEdit_4.setReadOnly(True)
        self.lineEdit_4.setObjectName("lineEdit_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 993, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton_prob2.setText(_translate("MainWindow", "show"))
        self.label_origin.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_origin.setText(_translate("MainWindow", "choose picture"))
        self.label_prob1.setText(_translate("MainWindow", "TextLabel"))
        self.pushButton_RGB.setText(_translate("MainWindow", "RGB"))
        self.pushButton_HSI.setText(_translate("MainWindow", "HSI"))
        self.pushButton_CMY.setText(_translate("MainWindow", "CMY"))
        self.pushButton_XYZ.setText(_translate("MainWindow", "XYZ"))
        self.pushButton_Lab.setText(_translate("MainWindow", "Lab"))
        self.pushButton_YUV.setText(_translate("MainWindow", "YUV"))
        self.label_gray.setText(_translate("MainWindow", "TextLabel"))
        self.label_color.setText(_translate("MainWindow", "TextLabel"))
        self.label_colorbar.setText(_translate("MainWindow", "TextLabel"))
        self.lineEdit.setText(_translate("MainWindow", "0"))
        self.label_kmean.setText(_translate("MainWindow", "TextLabel"))
        self.spinBox_k.setPrefix(_translate("MainWindow", "K: "))
        self.pushButton_kmean.setText(_translate("MainWindow", "kmean"))
        self.spinBox_level.setPrefix(_translate("MainWindow", "level: "))
        self.lineEdit_4.setText(_translate("MainWindow", "colormap"))

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
