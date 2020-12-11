from scipy.signal import *
import numpy as np
from PIL import Image


if __name__ == '__main__':
    im = Image.open('data/strawberries2.gif') 
    im = im.convert('RGB')
    im.save('data/strawberries2.jpg')