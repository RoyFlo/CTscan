from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from skimage.io import imread
import skimage.transform as skt


class GenerateCM(QMainWindow):
    def __init__(self, parent=None):
        super(GenerateCM, self).__init__()
        self.CM = np.zeros((10,10))
        #######################################
        self.centralWidget = QWidget()
        self.setCentralWidget(self.centralWidget)
        self.mainHBOX_param_scene = QHBoxLayout()

        #CM view
        layout_plot = QHBoxLayout()
        self.loaded_plot = CMViewer(self)
        self.loaded_plot.setMinimumHeight(200)
        self.loaded_plot.update()
        layout_plot.addWidget(self.loaded_plot)

        self.mainHBOX_param_scene.addLayout(layout_plot)
        self.centralWidget.setLayout(self.mainHBOX_param_scene)
        self.setWindowTitle("TITLE")
        self.setGeometry(400,400,1024,768)

class CMViewer(QGraphicsView):
    def __init__(self, parent=None):
        super(CMViewer, self).__init__(parent)
        self.parent=parent
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.canvas.setGeometry(0, 0, 1600, 500 )
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        self.canvas.show()

    def update(self):

        image = imread("Image1.png", as_gray=True)
        image = skt.rescale(image, scale=0.4, mode='reflect', multichannel=False)
        theta = np.linspace(0., 180., max(image.shape), endpoint=False)
        sinogram = skt.radon(image, theta=theta, circle=True)

        self.connectivityMat = self.parent.CM
        self.figure.clear()
        self.axes=self.figure.add_subplot(2,2,1)
        self.axes2=self.figure.add_subplot(2,2,2)
        im = self.axes.imshow(image, cmap=plt.cm.Greys_r)
        im2 = self.axes2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
        #self.figure.colorbar(im)
        self.figure.colorbar(im2)
        self.axes.axis()
        self.axes2.axis()
        self.axes.set_title('Original Image')
        self.axes2.set_title('Radon transform\n(Sinogram)')
        self.axes2.set_xlabel('Projection angle (deg)')
        self.axes2.set_ylabel('Projection position (pixels)')

        self.axes3 = self.figure.add_subplot(2, 2, 3)
        self.axes4 = self.figure.add_subplot(2, 2, 4)

        reconstruction_fbp = skt.iradon(sinogram, theta=theta, circle=True)
        error = reconstruction_fbp - image
        print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error ** 2)))

        imkwargs = dict(vmin=-0.2, vmax=0.2)
        #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4.5), sharex=True, sharey=True)
        #ax1.set_title("Reconstruction\nFiltered back projection")
        self.axes3.axis()
        self.axes4.axis()
        self.axes3.set_title('Reconstruction\nFiltered back porjection')

        im3 = self.axes3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
        #ax2.set_title("Reconstruction error\nFiltered back projection")
        im4 = self.axes4.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
        self.axes4.set_title('Reconstruction error\nFiltered back projection')


        self.canvas.draw()
        self.canvas.show()

def main():
    app = QApplication(sys.argv)
    ex = GenerateCM(app )
    ex.show()
    sys.exit(app.exec_( ))


if __name__ == '__main__':
    main()