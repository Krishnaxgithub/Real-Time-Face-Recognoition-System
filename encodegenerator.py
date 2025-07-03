import cv2
import face_recognition
import pickle
import os

# Importing the mode images into a list
folderPath = 'images'
PathList = os.listdir(folderPath)
print(PathList)

imgList = []
studentsIds = []
for path in PathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))

    studentsIds.append(os.path.splitext(path)[0])

    # print(path)
    # print(os.path.splitext(path)[0])

print(studentsIds)    


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentsIds]
print("Encoding Completed!")


file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()