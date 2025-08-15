import cv2
import cv2.data

img = cv2.imread("images/profationalimg.jpeg")

detect_face = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

faccs = detect_face.detectMultiScale(img, 1.2, 4)

for x, y, w, h in faccs:
    cv2.rectangle(img, (x, y), (x+w, y+h), (56, 244, 23), 3)

cv2.imshow("image", img)

cv2.waitKey(0)