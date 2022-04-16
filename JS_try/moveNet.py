import cv2

def get_web():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        cv2.imshow('MoveNet Lightning',frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


# get_web()

