import cv2
import numpy as np
import sys
import os


def detect_red(bgr_file):
    bgr_img = hsv_img = lower_red_hue = upper_red_hue = cv2.imread(bgr_file) # init. new imgs to be same as bgr_img

    """ convert bgr_img to an hsv_img """
    cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV, hsv_img)
    """ threshold hsv_img, keeping only red pixels """
    lower_red_hue = cv2.inRange(hsv_img, np.array([0, 100, 100]), np.array([10, 255, 255]), lower_red_hue)
    upper_red_hue = cv2.inRange(hsv_img, np.array([160, 100, 100]), np.array([179, 255, 255]), upper_red_hue)

    """ combining lower and upper red hues and smoothing out result """
    red_hue_combined = cv2.imread(bgr_file)
    red_hue_combined = cv2.addWeighted(lower_red_hue, 1.0, upper_red_hue, 1.0, 0.0, red_hue_combined)
    cv2.GaussianBlur(red_hue_combined, (3, 3), 2, red_hue_combined, 2)


    """ save images """
    img_file_dir = os.path.splitext(bgr_file)[0] + '_select-red-text'
    if not os.path.exists(img_file_dir):
        os.makedirs(img_file_dir)

    cv2.imshow('hsv_img', hsv_img)
    cv2.imwrite(img_file_dir + '/hsv_img.jpg', hsv_img)

    cv2.imshow('lower_red_hue', lower_red_hue)
    cv2.imwrite(img_file_dir + '/lower_red_hue.jpg', lower_red_hue)

    cv2.imshow('upper_red_hue', upper_red_hue)
    cv2.imwrite(img_file_dir + '/upper_red_hue.jpg', upper_red_hue)

    cv2.imshow('red_hue_combined', red_hue_combined)
    cv2.imwrite(img_file_dir + '/red_hue_combined.jpg', red_hue_combined)

    cv2.waitKey()


def main():
    img_file = sys.argv[1]
    detect_red(img_file)

if __name__ == "__main__":
    main()

