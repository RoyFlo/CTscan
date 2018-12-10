from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.io import imread
import skimage.transform as skt

window = Tk()
window.geometry("500x550")

fig = Figure(figsize=(5, 5))
canvas = FigureCanvasTkAgg(fig, master=window)


def plot():
    image = imread("Image1.png", as_gray=True)
    image = skt.rescale(image, scale=0.4, mode='reflect', multichannel=False)

    a1 = fig.add_subplot(221)
    a2 = fig.add_subplot(222)

    a1.imshow(image, cmap=plt.cm.Greys_r)
    a1.set_title("Original")

    theta = np.linspace(0., 180., max(image.shape), endpoint=False)
    sinogram = skt.radon(image, theta=theta, circle=True)

    a2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    a2.set_title("Radon transform\n(Sinogram)")
    a2.set_xlabel("Projection angle (deg)")
    a2.set_ylabel("Projection position (pixels)")

    a3 = fig.add_subplot(223)
    a4 = fig.add_subplot(224)

    reconstruction_fbp = skt.iradon(sinogram, theta=theta, circle=True)
    error = reconstruction_fbp - image
    print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error ** 2)))
    imkwargs = dict(vmin=-0.2, vmax=0.2)

    a3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    a3.set_title('Reconstruction\nFiltered back porjection')

    a4.imshow(reconstruction_fbp - image, cmap=plt.cm.Greys_r, **imkwargs)
    a4.set_title('Reconstruction error\nFiltered back projection')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(row=1, columnspan=5)
    canvas.draw()
    print("Plotting")


def clear():
    print("clearing")


button1 = Button(window, text="Image 1", command=plot)
button2 = Button(window, text="Image 2", command=plot)
button3 = Button(window, text="Image 3", command=plot)
button4 = Button(window, text="Image 4", command=plot)
button5 = Button(window, text="Image 5", command=plot)
button6 = Button(window, text="CLEAR", command=clear)

button1.grid(row=0, column=0)
button2.grid(row=0, column=1)
button3.grid(row=0, column=2)
button4.grid(row=0, column=3)
button5.grid(row=0, column=4)
button6.grid(row=5, columnspan=5)

window.mainloop()