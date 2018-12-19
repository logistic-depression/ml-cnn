import PIL, os
from PIL import Image

def rotate_img(input_name, output_name, rot):
    img = Image.open(input_name)
    img.rotate(rot).save(output_name)

def flip_img(input_name, output_name): 
    img = Image.open(input_name) 
    img.transpose(Image.FLIP_LEFT_RIGHT).save(output_name) 

filename_g = "training/groundtruth/"
filename_i = "training/images/"

# first flip all images
for i in range(1, 101):
    imageid = "satImage_%.3d" % i
    imageid_out = "satImage_%.3d" % (i+100)
    input_name_g = filename_g + imageid + ".png"
    output_name_g = filename_g + imageid_out + ".png"
    input_name_i = filename_i + imageid + ".png"
    output_name_i = filename_i + imageid_out + ".png"
    flip_img(input_name_g, output_name_g) 
    flip_img(input_name_i, output_name_i)

# we now have 200 images, we do the rotations:
for i in range(i, 201):
    imageid = "satImage_%.3d" % i
    imageid_90 = "satImage_%.3d" % (i+200)
    imageid_180 = "satImage_%.3d" % (i+400)
    imageid_270 = "satImage_%.3d" % (i+600)
    imageid_outs = [imageid_90, imageid_180, imageid_270]

    input_name_g = filename_g + imageid + ".png"
    output_names_g = [filename_g + imageid_out + ".png" for imageid_out in imageid_outs]
    input_name_i = filename_i + imageid + ".png"
    output_names_i = [filename_i + imageid_out + ".png" for imageid_out in imageid_outs]

    rotations = [90, 180, 270]
    for i in range(len(rotations)):
        rotate_img(input_name_g, output_names_g[i], rotations[i])
        rotate_img(input_name_i, output_names_i[i], rotations[i])
