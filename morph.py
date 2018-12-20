import PIL, os
from PIL import Image
import cv2
import numpy as np
import skimage as skimage

# BY RUNNING THIS FILE, ALL IMAGES IN filename ARE MORPHOLOGICALLY CLOSED
# WITH A KERNEL SIZE OF 17x17, IF THEY HAVE THE RIGHT FILENAME (imageid)

def closeim(input_name, output_name, n):
    kernel = np.ones((n,n),np.uint8)
    img = np.asarray(Image.open(input_name))
    img = img[:,400:800,0]
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    seeds = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    skimage.morphology.reconstruction(seeds, img, method='dilation')
    Image.fromarray(np.uint8(img)).save(output_name)

filename = "predictions_training-v2/predictions_training/"

for i in range(1, 801):
    try:
        imageid = "prediction_%.d" % i
        imageid_out = "morphoutputImage_%.3d" % i
        input_name = filename + imageid + ".png"
        output_name = filename + imageid_out + ".png"
        closeim(input_name, output_name, 17)
    except:
	    1+1