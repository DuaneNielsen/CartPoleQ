import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

img=mpimg.imread('stinkbug.png')

#plt.interactive(False)
imgplot = plt.imshow(img)
plt.show()
#lum_img = img[:,:,0]

#plt.imshow(lum_img)