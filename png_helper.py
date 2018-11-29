import glob
import numpy as np
import imageio

def load_pngs(filenames):
    filelist = glob.glob(filenames)
    return np.array([np.array(imageio.imread(fname)) for fname in filelist])

def load_data():
    train_x = load_pngs("training/images/satImage_???.png")
    train_y = load_pngs("training/groundtruth/satImage_???.png")
    test_x = load_pngs("test_set_images/test_*/test_*.png")
    return (train_x, train_y), test_x