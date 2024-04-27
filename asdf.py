import cv2
import numpy as np

image = cv2.imread('img1.jpg')
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

(humans, _) = hog.detectMultiScale(image, winStride=(6, 6), padding=(8, 8), scale=1.075)
print("Jumlah orang terdeteksi:", len(humans))

### Sampai sini kode untuk mendata jumlah orang yang terdeteksi ###
###################################################################
### Kode selanjutnya untuk memvisualisasikan pendeteksian orang ###

for (x, y, w, h) in humans:
    pad_w, pad_h = int(0.15 * w), int(0.01 * h)
    cv2.rectangle(image, (x + pad_w, y + pad_h), (x + w - pad_w, y + h - pad_h), (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
#cv2.destroyAllWindows()
