import numpy as np
from matplotlib import pyplot as plt

def showim(*imgs):
	numimgs = len(imgs)
	Nx = int(np.round(np.sqrt(numimgs)))
	Ny = int(np.ceil(np.sqrt(numimgs)))
	format = str(Nx)+str(Ny)
	for k in range(0,numimgs):
		plt.subplot(format+str(k+1))
		if len(imgs[k].shape)==3:
			plt.imshow(imgs[k])
		else:
			plt.imshow(imgs[k], cmap='gray')
	plt.show()