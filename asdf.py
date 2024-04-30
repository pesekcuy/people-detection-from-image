import cv2
import numpy as np

image = cv2.imread('Untitled.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(humans, _) = hog.detectMultiScale(image, winStride=(12, 12), padding=(24, 24), scale=1.05)
print("Human detected:", len(humans))

### Sampai sini kode untuk mendata jumlah orang yang terdeteksi ###
###################################################################
### Kode selanjutnya untuk memvisualisasikan pendeteksian orang ###

for (x, y, w, h) in humans:
    pad_w, pad_h = int(0.15 * w), int(0.01 * h)
    cv2.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (255, 255, 255), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
