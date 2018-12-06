import glob
import numpy as np
import imageio

def load_pngs(namepattern):
    filelist = sorted(glob.glob(namepattern))
    return np.array([np.array(imageio.imread(fname)) for fname in filelist])

def load_data():
    x_train = load_pngs("training/images/satImage_???.png")
    y_train = load_pngs("training/groundtruth/satImage_???.png")
    x_test = load_pngs("test_set_images/test_*/test_*.png")
    return (x_train, y_train), x_test