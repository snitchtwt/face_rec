import cv2
import numpy as np
import face_recognition
import os  
from datetime import datetime


path = '/Users/snitch/Desktop/Second Yr Proj/img' 
images = []                     
Student_List = []               
myList = os.listdir(path)       #importing the images 
print("**********// Loading all Student Data //**********")
print(myList)

# Reading all the images in python using opencv
for cu_img in myList:
    temp_ls = cv2.imread(f'{path}/{cu_img}')
    images.append(temp_ls)
    Student_List.append(os.path.splitext(cu_img)[0])
print("**********// Preparing Student List //**********")
print(Student_List)
print("**********// Please wait for further computation //**********")

# Face recogniton module is based on DLIB so it check for 128 extream characteristic in the face 
def faceEncodings(images): 
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #converting face data from BGR(default of OpenCV) to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
# The list encodeList has individaul arrays for each element in student list with a size of 128 encodings 
# The algo used in this is called the HOG(histogram of Oriented Gradients) tranformation
def attendance(name):
    with open('att.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
print("")
print("**********// Ready to take Attendance //**********")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB) #converting face data from BGR(default of OpenCV) to RGB

    facesCurrentFrame = face_recognition.face_locations(faces) #searches for faces
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame) #enocdings from cam as well as the data 

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = Student_List[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows