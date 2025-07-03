# webcam, graphics, encoding generator, face recognition, database setup, add to the database, add images to database, real time database update, limit the no.of attendence per day


import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime



# Webcam setup
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Load background image
imgbackground = cv2.imread('resources/background.png')

# Load mode images if needed (optional)
foldermodePath = 'resources/modes'
modePathList = os.listdir(foldermodePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(foldermodePath, path)))

# Load face encodings and IDs
print("Loading Encode File...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentsIds = encodeListKnownWithIds
print("Encode File Loaded!")

# Resize image keeping aspect ratio and pad with black to fit width x height
def resize_with_aspect_ratio(image, width, height):
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y_offset = (height - new_h) // 2
    x_offset = (width - new_w) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from webcam")
        break

    # Mirror the webcam image
    img = cv2.flip(img, 1)

    # Resize and fit webcam image inside background at (55, 162) with size 640x480
    webcam_width, webcam_height = 640, 480
    img_fitted = resize_with_aspect_ratio(img, webcam_width, webcam_height)

    # Copy the resized webcam into background
    imgbackground[162:162+webcam_height, 55:55+webcam_width] = img_fitted
    imgbackground[44:44 + 633, 808:808 + 414] = imgModeList[3]


    # Prepare small frame for face detection (scale down by 4)
    imgS = cv2.resize(img_fitted, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces and encode
    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

        print("matches", matches)
        print("faceDis", faceDis)

        if len(faceDis) > 0:
            matchIndex = np.argmin(faceDis)
            
            if matches[matchIndex]:
                # Scale face location back to img_fitted size (multiply by 4)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                # Calculate bbox position relative to full background
                bbox_x = 55 + x1
                bbox_y = 162 + y1
                bbox_w = x2 - x1
                bbox_h = y2 - y1

                # Draw bounding box with cornerRect on imgbackground
                imgbackground = cvzone.cornerRect(imgbackground, (bbox_x, bbox_y, bbox_w, bbox_h), rt=0)

                # Add student ID text above bbox
                cv2.putText(imgbackground, f"ID: {studentsIds[matchIndex]}", (bbox_x, bbox_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the combined output window
    cv2.imshow("Face Attendance", imgbackground)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
