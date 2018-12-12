#Rogelio Flores - 125726
#Rafael Sanchez - 1568678


from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from skimage.io import imread
import skimage.transform as skt

window = Tk()
window.geometry("700x820")
window.title("CT Scanner")

Label(window, text="1. Select an image from the list below.", font=("Ariel", 12), fg="blue").grid(row=0)
Label(window, text="2. Select the scale for the output.\n(1/n of original resolution)", font=("Ariel", 12), fg="blue").grid(row=0, column=1)
Label(window, text="3. Hit Run Simulation", font=("Ariel", 12), fg="blue").grid(row=0, column=3)

this = "Image1.png"
scale = 4
print("Default image is " + this + " and scanner resolution will be \n\t1/" + str(scale) + " the resolution of original.")

def iValue(value):
    global this
    this = value+".png"
    print("You have selected " + this)

def sValue(value):
    global scale
    scale = value
    print(scale)

iList = ["Image1", "Image2", "Image3", "Image4", "Image5", "SheppLogan_Phantom"]
sList = [1,2,3,4,5,6,7,8,9,10]
var1=StringVar()
var1.set("Image1")
var2=StringVar()
var2.set(4)
set1 = OptionMenu(window, var1, *iList, command=iValue)
set1.configure(font=("Ariel"))
set1.grid(row=1, column=0)
set2 = OptionMenu(window, var2, *sList, command=sValue)
set2.configure(font=("Ariel"))
set2.grid(row=1, column=1)

fig = Figure(figsize=(7, 7))
canvas = FigureCanvasTkAgg(fig, master=window)

def plot():
    print("\nSimulating using " + this + " using Filtered Back Projection (FBP) for reconstruction")
    image = imread(this, as_gray=True)
    image2 = skt.rescale(image, scale=1/scale, mode='reflect', multichannel=False)
    scaledres = int((image.shape[0])/scale)
    print("Scanner resolution is set to " + str(scaledres) + " x " + str(scaledres))

    a1 = fig.add_subplot(221)
    a2 = fig.add_subplot(222)

    a1.imshow(image, cmap=plt.cm.Greys_r)
    a1.set_title("Original Image")

    numofSlices = max(image2.shape)

    print("Creating sinogram...")
    #theta = np.linspace(0., degree, max(image2.shape), endpoint=False)
    theta = np.linspace(0., 180, numofSlices, endpoint=False)
    sinogram = skt.radon(image2, theta=theta, circle=True)
    print("Sinogram complete")

    a2.imshow(sinogram, cmap=plt.cm.Greys_r, extent=(0, 180, 0, sinogram.shape[0]), aspect='auto')
    a2.set_title("Radon transform\n(Sinogram)")
    a2.set_xlabel("Projection angle (deg)")
    a2.set_ylabel("Projection position (pixels)")

    a3 = fig.add_subplot(223)
    a4 = fig.add_subplot(224)

    print("Image being reconstructed")
    reconstruction_fbp = skt.iradon(sinogram, theta=theta, circle=True)
    print("Reconstruction complete.")

    error = reconstruction_fbp - image2
    print('FBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error ** 2)))
    rerror = 'Scanner resolution is set to ' + str(scaledres) + ' x ' + str(scaledres) + '\nFBP rms reconstruction error: %.3g' % np.sqrt(np.mean(error ** 2))
    imkwargs = dict(vmin=-0.2, vmax=0.2)

    a3.imshow(reconstruction_fbp, cmap=plt.cm.Greys_r)
    a3.set_title('Reconstruction with\nFiltered back projection')

    a4.imshow(reconstruction_fbp - image2, cmap=plt.cm.Greys_r, **imkwargs)
    a4.set_title('Reconstruction error\ndifference')

    fig.tight_layout()
    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.get_tk_widget().grid(row=2, columnspan=5)
    Label(window, text=rerror, font=("Ariel", 10), fg="red").grid(row=3, sticky=SE)
    canvas.draw()

button1 = Button(window, text="Run Simulation", bg="red", command=plot)
button1.grid(row=1, column=3)

window.mainloop()