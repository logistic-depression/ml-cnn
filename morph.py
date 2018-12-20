import PIL, os
from PIL import Image
import cv2
import numpy as np

# BY RUNNING THIS FILE, ALL IMAGES IN filename ARE MORPHOLOGICALLY CLOSED
# WITH A KERNEL SIZE OF 17x17, IF THEY HAVE THE RIGHT FILENAME (imageid)

def closeim(input_name, output_name, n):
    kernel = np.ones((n,n),np.uint8)
    img = np.asarray(Image.open(input_name))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    Image.fromarray(np.uint8(img)).save(output_name)

filename = "morph/"

for i in range(1, 101):
    try:
        imageid = "outputImage_%.3d" % i
        imageid_out = "morphoutputImage_%.3d" % i
        input_name = filename + imageid + ".png"
        output_name = filename + imageid_out + ".png"
        closeim(input_name, output_name, 17)
    except:
        print("Image ",i," was skipped.")