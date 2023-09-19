import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

data_path = 'C:/Users/Subhanshu Raj/OneDrive/Desktop/PROJECT/image/'
onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path,f))]

Training_Data, Labels = [], []

for i, files in enumerate(onlyfiles):
    image_path = data_path + onlyfiles[i]
    images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    Training_Data.append(np.asarray(images, dtype=np.uint8))
    Labels.append(i)

Labels = np.asarray(Labels, dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

# model.train(np.asarray(Training_Data), np.asarray(Labels))

print("Dataset Model Training Complete!!!!!")

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def face_detector(img, size = 0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return img,[]

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200,200))

    return img,roi

cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()

    image, face = face_detector(frame)

    try:
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model.predict(face)

        if result[1] < 500:
            confidence = int(100*(1-(result[1])/300))



        if confidence > 82:
            cv2.putText(image, "SUBHANSHU", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Face Cropper', image)

        else:
            cv2.putText(image, "Unknown", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Face Cropper', image)


    except:
        cv2.putText(image, "Subhanshu", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Face Cropper', image)
        pass

    if cv2.waitKey(1)==13:
        break


cap.release()
cv2.destroyAllWindows()

#
# import cv2
# import numpy as np
# import os
#
# recognizer = cv2.face.LBPHFaceRecognizer_create()
# recognizer.read('trainer/trainer.yml')
# # recognizer.read('C:/Users/lenovo/Pictures/dataset/')
# cascadePath = "C:/Users/lenovo/Downloads/haarcascade_frontalface_default.xml"
# faceCascade = cv2.CascadeClassifier(cascadePath);
# font = cv2.FONT_HERSHEY_SIMPLEX
# # iniciate id counter
# id = 0
# # names related to ids: example ==> Marcelo: id=1,  etc
# names = ['None', 'ROZY', 'SRK', 'ALONE MASK', 'Z', 'W']
# # Initialize and start realtime video capture
# cam = cv2.VideoCapture(0)
# cam.set(3, 640)  # set video widht
# cam.set(4, 480)  # set video height
# # Define min window size to be recognized as a face
# minW = 0.1 * cam.get(3)
# minH = 0.1 * cam.get(4)
# while True:
#     ret, img = cam.read()
#     img = cv2.flip(img, -1)  # Flip vertically
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     faces = faceCascade.detectMultiScale(
#         gray,
#         scaleFactor=1.2,
#         minNeighbors=5,
#         minSize=(int(minW), int(minH)),
#     )
#     for (x, y, w, h) in faces:
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         id, confidence = recognizer.predict(gray[y:y + h, x:x + w])
#
#         # If confidence is less them 100 ==> "0" : perfect match
#         if (confidence < 100):
#             id = names[id]
#             confidence = "  {0}%".format(round(100 - confidence))
#         else:
#             id = "unknown"
#             confidence = "  {0}%".format(round(100 - confidence))
#
#         cv2.putText(
#             img,
#             str(id),
#             (x + 5, y - 5),
#             font,
#             1,
#             (255, 255, 255),
#             2
#         )
#         cv2.putText(
#             img,
#             str(confidence),
#             (x + 5, y + h - 5),
#             font,
#             1,
#             (255, 255, 0),
#             1
#         )
#
#     cv2.imshow('camera', img)
#     k = cv2.waitKey(10) & 0xff  # Press 'ESC' for exiting video
#     if k == 27:
#         break
# # Do a bit of cleanup
# print("\n [INFO] Exiting Program and cleanup stuff")
# cam.release()
# cv2.destroyAllWindows()