import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import winsound
import time
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
EYE_AR_THRESH = 0.25  
EYE_AR_CONSEC_FRAMES = 3 
BLINK_COUNT = 0
ALARM_DURATION = 3
ALARM_INTERVAL = 1.0
eye_closed_start_time = None
last_alarm_time = None
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
def beep_alarm():
    frequency = 2500  
    duration = 1000  
    winsound.Beep(frequency, duration)
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break       
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])        
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]        
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0        
        if ear < EYE_AR_THRESH:
            if eye_closed_start_time is None:
                eye_closed_start_time = time.time()
            else:
                closed_duration = time.time() - eye_closed_start_time
                if closed_duration >= ALARM_DURATION:                
                    if last_alarm_time is None or (time.time() - last_alarm_time) >= ALARM_INTERVAL:
                        beep_alarm()
                        last_alarm_time = time.time()
                    cv2.putText(frame, "ALERT! EYES CLOSED TOO LONG!", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            if eye_closed_start_time is not None:
                closed_duration = time.time() - eye_closed_start_time
                if closed_duration >= EYE_AR_CONSEC_FRAMES/30: 
                    BLINK_COUNT += 1
                eye_closed_start_time = None
                last_alarm_time = None            
        cv2.drawContours(frame, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)
        cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (255, 0, 0), 2)        
    cv2.putText(frame, f"Blinks: {BLINK_COUNT}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"EAR: {ear:.2f}", (frame.shape[1]-150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)    
    cv2.imshow("Eye Blink Detection", frame)    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()