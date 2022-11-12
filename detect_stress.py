from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import time
import dlib
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

ar_thresh = 0.3
eye_ar_consec_frame = 5
counter = 0
total = 0


def distanceEyebrows(leftEye,rightEye):
    global distance
    distEuc = dist.euclidean(leftEye,rightEye)
    distance.append(int(distEuc))
    return distEuc

def detectEmotions (faces,frame):
    global emotionClassifierModel
    EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
    x,y,w,h = face_utils.rect_to_bb(faces)
    frame = frame[y:y+h,x:x+w]
    roi = cv2.resize(frame,(64,64))
    roi = roi.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi,axis=0)
    preds = emotionClassifierModel.predict(roi)[0]
    emotion_probability = np.max(preds)
    label = EMOTIONS[preds.argmax()]
    if label in ['angry','scared','sad']:
        label = 'Stressed'
    else:
        label = 'Not Stressed'
    return label

def normalizeValues (distance,disp):
    normalizedValue = abs(disp - np.min(distance))/abs(np.max(distance) - np.min(distance))
    stressValue = np.exp(-(normalizedValue))
    if stressValue>=60:
        return stressValue,"High Stress"
    else:
        return stressValue,"Low Stress"

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    eye_opening_ratio = (A + B) / (2.0 * C)
    return eye_opening_ratio


faceDetector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
emotionClassifierModel = load_model("_mini_XCEPTION.102-0.66.hdf5", compile=False)
cap = cv2.VideoCapture(0)
distance = []

while(True):

    _,frame = cap.read()
    frame = cv2.flip(frame,1)
    frame = imutils.resize(frame, width=500,height=500)
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    (rightEyeBegin, rightEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eyebrow"]
    (leftEyeBegin, leftEyeEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eyebrow"]
    (mouthStart, mouthEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    (lBegin, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rBegin, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    faceDetection = faceDetector(gray,0)
    for face in faceDetection:
        detectedEmotion = detectEmotions(face, gray)
        cv2.putText(frame, detectedEmotion, (20,20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        faceShape = predictor(frame, face)
        faceShape = face_utils.shape_to_np(faceShape)
        left_eye = faceShape[lBegin:lEnd]
        right_eye = faceShape[rBegin:rEnd]
        leftEyeBrow = faceShape[leftEyeBegin:leftEyeEnd]
        rightEyeBrow = faceShape[rightEyeBegin:rightEyeEnd]
        mouth = faceShape[mouthStart:mouthEnd]
        leftEyeBrowHull = cv2.convexHull(leftEyeBrow)
        rightEyeBrowHull = cv2.convexHull(rightEyeBrow)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeBrowHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeBrowHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouth], -1, (0, 255, 0), 1)
        left_eye_Ear = eye_aspect_ratio(left_eye)
        right_eye_Ear = eye_aspect_ratio(right_eye)
        avg_Ear = (left_eye_Ear + right_eye_Ear)/2.0
        if avg_Ear<ar_thresh:
            counter+=1
        else:
            if counter>eye_ar_consec_frame:
                total+= 1
            counter = 0
        distq = distanceEyebrows(leftEyeBrow[-1],rightEyeBrow[0])
        stressValue,stressLabel = normalizeValues(distance,distq)
        cv2.putText(frame,"Stress Level: {}".format(str(int(stressValue*100))),(20,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame,"Approx. Stress Level: {}".format(str(stressLabel)),(20,60),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame,"Blink Count: {}".format(str(int(total))),(20,80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
plt.plot(range(len(distance)),distance,'ro')
plt.title("Stress Levels")
plt.show()      

