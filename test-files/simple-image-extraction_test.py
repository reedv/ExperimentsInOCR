
from matplotlib import pyplot as plot
import numpy as np
from PIL import Image
import pytesseract
import sys


def text_ex(img_file):
    img = Image.open(img_file)
    imgArray = np.asarray(img)
    print "Detected text:"
    print pytesseract.image_to_string(img)

    imgPlot = plot.imshow(img)
    plot.show()

def main():
    text_ex(sys.argv[1])

if __name__ == "__main__":
    main()
