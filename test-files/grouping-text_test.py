import cv2
import sys
import os


def captch_ex(img_file):
    img = cv2.imread(img_file)

    """ removing noisy portion """
    img_final = cv2.imread(img_file)
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img2gray', img2gray)

    ret, mask = cv2.threshold(img2gray, 180, 255, cv2.THRESH_BINARY)
    cv2.imshow('img_mask', mask)
    image_final = cv2.bitwise_and(img2gray , img2gray , mask=mask)
    cv2.imshow('img_final', img_final)

    ret, new_img = cv2.threshold(image_final, 180 , 255, cv2.THRESH_BINARY)  # black text: cv.THRESH_BINARY_INV, white text: cv.THRESH_BINARY
    cv2.imshow('new_img', new_img)


    # to manipulate the orientation of dilution ,
    # large x means horizonatally dilating more,
    # large y means vertically dilating more
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3 , 3))    
    # dilate, more the iteration more the dilation
    dilated = cv2.dilate(new_img,kernel, iterations = 9) 
    _, contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # get contours (unpacks 3 values)

    """ add rectangles around all detected contours """
    crop_index = 1
    for contour in contours:
        # get rectangle bounding contour
        [x,y,w,h] = cv2.boundingRect(contour)

        #Don't plot small false positives that aren't text (can be adjusted)
        if w < 30 and h<30:
            continue

        # draw rectangle around contour on original image
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 2)


        """ you can crop image and send to OCR, false detected will return no text """
        cropped = img_final[y :y +  h , x : x + w]

        img_file_dir = os.path.splitext(img_file)[0] + '-group-crops'
        if not os.path.exists(img_file_dir):
            os.makedirs(img_file_dir)

        s = img_file_dir + '/crop_' + str(crop_index) + '.jpg'
        cv2.imwrite(s , cropped)
        crop_index = crop_index + 1


    """ display original image with added contours """
    cv2.imshow('captcha_result - original img' , img)
    cv2.waitKey()


def main():
    img_file = sys.argv[1]
    captch_ex(img_file)

if __name__ == "__main__":
    main()

