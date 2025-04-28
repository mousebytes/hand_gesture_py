import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5,min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)


def is_finger_extended(tip, pip):
    return tip.y < pip.y

def check_peace_sign(landmarks):
    index_extended = is_finger_extended(landmarks[8], landmarks[6])
    middle_extended = is_finger_extended(landmarks[12], landmarks[10])
    ring_folded = not is_finger_extended(landmarks[16], landmarks[14])
    pinky_folded = not is_finger_extended(landmarks[20], landmarks[18])

    if index_extended and middle_extended and ring_folded and pinky_folded:
        return True
    return False

def check_middle_finger(landmarks):
    middle_extended = is_finger_extended(landmarks[12], landmarks[10])
    index_folded = not is_finger_extended(landmarks[8], landmarks[6])
    ring_folded = not is_finger_extended(landmarks[16], landmarks[14])
    pinky_folded = not is_finger_extended(landmarks[20], landmarks[18])
    thumb_folded = landmarks[4].x > landmarks[3].x  # Assuming palm facing camera

    if middle_extended and index_folded and ring_folded and pinky_folded and thumb_folded:
        return True
    return False



peace_sign = "Peace Sign"
middle_finger = "Middle Finger"

while(cap.isOpened()):
    ret,frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)

    rgb_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(rgb_frame)




    if(results.multi_hand_landmarks):
        for landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame,landmarks,mp_hands.HAND_CONNECTIONS)

            #check peace sign
            if(check_peace_sign(landmarks.landmark)):
                cv2.putText(frame,peace_sign,(50,50),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3,cv2.LINE_AA)

            #check middle finger
            if(check_middle_finger(landmarks.landmark)):
                cv2.putText(frame,middle_finger,(50,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3,cv2.LINE_AA)

    cv2.imshow("HAnd gesture recognition", frame)

    #exit when q pressed
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()