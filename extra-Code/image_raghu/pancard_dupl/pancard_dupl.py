import os
import cv2
import numpy as np
image_folder = os.path.join('data/')
image_files = [_ for _ in os.listdir(image_folder) if _.endswith('jpg')]

duplicate_files = []
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
for file_org in image_files:
    if not file_org in duplicate_files:
        image_org = cv2.imread(os.path.join(image_folder,file_org))
        gray = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=3,
            minSize=(30, 30)
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(image_org, (x, y), (x + w, y + h), (255, 0, 0), 2)
        im1 = image_org[y:y + h, x:x + w]
        #cv2.imwrite(str(w) + str(h) + '_faces.jpg', im1)
        #cv2.imshow('Img', im1)
        #cv2.waitKey(0)
        pix_mean1=cv2.mean(im1,mask=None)
        for file_check in image_files:
            if file_check != file_org:
                image_check = cv2.imread(os.path.join(image_folder, file_check))
                gray_check = cv2.cvtColor(image_org, cv2.COLOR_BGR2GRAY)
                faces_check = faceCascade.detectMultiScale(
                    gray_check,
                    scaleFactor=1.3,
                    minNeighbors=3,
                    minSize=(30, 30)
                )
                for (x1, y1, w1, h1) in faces_check:
                    cv2.rectangle(image_check, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 2)
                im1_check = image_check[y1:y1 + h1, x1:x1 + w1]
                # cv2.imwrite(str(w) + str(h) + '_faces.jpg', im1)
                #cv2.imshow('Img', im1_check)
                #cv2.waitKey(0)
                pix_mean2 = cv2.mean(im1_check,mask=None)
                p1=np.mean(pix_mean1)
                p2=np.mean(pix_mean2)
                ab=abs(p1-p2)
                if ab <= 0.5:
                    duplicate_files.append((file_org))
                    duplicate_files.append((file_check))
                    image_duplic = cv2.imread(os.path.join(image_folder, file_org))
                    cv2.imshow('dupli',image_duplic)
                    cv2.waitKey(0)
print(list(dict.fromkeys(duplicate_files)))