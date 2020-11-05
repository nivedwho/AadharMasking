import cv2
import numpy 
import pytesseract as pt
from pytesseract import Output
import re

#function for normalizing input image for better OCR
def preprocess(img):
    dilated_img = cv2.dilate(img, numpy.ones((7, 7), numpy.uint8))
    bg_img = cv2.medianBlur(dilated_img, 15)
    diff_img = 255 - cv2.absdiff(img, bg_img)
    norm_img = diff_img.copy() 
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return thr_img

#function that rotates the input if nescessary
def imgRot(img):
    count=0
    img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
    count+=1
    return img,count

#the image is rotated back to original orientation after masking
def rotBack(img,count):
    for i in range(count):
        img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return img

#function to get Aadhaar Number from the input image
def getNum(img):
    text = pt.image_to_string(img)
    num = set()
    try:
        newlist = []
        for xx in text.split('\n'):
            newlist.append(xx)
        #filter out many small lines detected. We know that aadhar number will contain 12 digits with spaces in between, so the size will be obviously >12
        newlist = list(filter(lambda x: len(x) > 12, newlist))

        # We need only those lines that contains ONLY digits from 0-9. Regular Expression is used for checking this condition. 
        for no in newlist:
            if re.match("^[0-9 ]+$", no):
                num.add(no)                

    except Exception:
        pass

    #Convert set type to string
    num = " ".join(num)
    return num

#function that masks the image
def Mask(org,img, num):
    d = pt.image_to_data(img, output_type=Output.DICT, lang='eng')
    n_boxes = len(d['level'])

    #The detected Aadhar number is a string contaaining all 12 digits. We only need the first 4. 
    num = num.split()[0]
    
    #masking is performed on the original image copy, but ocr is performed on the preprocessed image.
    overlay = org.copy()

    for  i in range(n_boxes):
        text = d['text'][i]
        #text will contain all the words or digits present in the image but we need only the text containing the first 4 numbers of Aadhaar number 
        if text == num:
            # x,y marks the starting of the first 4 numbers and w,h stores width and height respectively  
            (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
            # x1,y1 marks the starting of the second 4 numbers and w1,h1 stores width and height respectively 
            (x1, y1, w1, h1) = (d['left'][i + 1], d['top'][i + 1], d['width'][i + 1], d['height'][i + 1])
            #Draw a rectangle in this position
            cv2.rectangle(overlay, (x, y), (x1+w1 , y1+h1), (70, 70, 70), -1)
            
    return overlay

#function that saves the masked images locally
def saveMask(img,count):
    img_name = "masked_{}.png".format(count)
    cv2.imwrite(img_name, img)
    print("Saved {}".format(img_name))
    count+=1
    return img_name, count

if __name__ == "__main__":
    imgList = ['img1.jpeg','img2.jpeg','img3.png']
    maskList = []
    p = 1

    for i in imgList:
        image = cv2.imread(i)
        #The output and input should look the same, for this purpose the input image is stored in org and is later used.
        org = image

        image = preprocess(image)
        
        c = 0
        num= ''

        #Checks if the input image is in correct orientation. This is done by rotating the image in loop and checking if Aadhar number can be detected each time. 
        while(c<4 and len(num)==0):
            image,count = imgRot(image)
            num = getNum(image)
            c+=1    
        #This message will be displayed if Aadhar number is not detected.
        if num == "":
            print("Sorry the given input image({}) cannot be masked.".format(i))
            break

        #The original input image should also be rotated in the similar manner, only then masking can be done properly.
        for i in range(count):
            org, count = imgRot(org)

        #The correctly oriented image can now be used for masking
        im = Mask(org, image,num)

        #The input and output should be same orientation. So we need to rotate back to original orientaion.
        img = rotBack(im,count)
        
        maskName, p = saveMask(img,p)
        maskList.append(maskName)

    #Displays all the masked images together.
    for i in maskList:
        image = cv2.imread(i)
        cv2.imshow('masked images', image)
        cv2.waitKey(0)