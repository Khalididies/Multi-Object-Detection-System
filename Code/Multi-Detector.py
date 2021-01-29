import cv2

# person 2
body_cascade = cv2.CascadeClassifier('haarcascade_fullbody.xml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# car 2
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')
license_cascade = cv2.CascadeClassifier('haarcascade_license.xml')

# street signs 3
sign_cascade = cv2.CascadeClassifier('haarcascade_TrafficSign.xml')
TrafficLight_cascade = cv2.CascadeClassifier('haarcascade_TrafficLight.xml')

# person 2
def detect_body(frame):
    body = body_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in body:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

def detect_face(frame):
    face = face_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in face:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

# car 2
def detect_car(frame):
    car = car_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in car:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

def detect_license(frame):
    license = license_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in license:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

# street signs 2
def detect_sign(frame):
    sign = sign_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in sign:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

def detect_TrafficLight(frame):
    TrafficLight = TrafficLight_cascade.detectMultiScale(frame, 1.15, 4)
    for (x, y, w, h) in TrafficLight:
        cv2.rectangle(frame, (x, y), (x+w,y+h), color=(0, 255, 0), thickness=2)
    return frame

def detection():
    #Video = cv2.VideoCapture('')
    Video = cv2.VideoCapture(1)
    while Video.isOpened():
        ret, frame = Video.read()
        frame = cv2.resize(frame, (640,480))
        controlkey = cv2.waitKey(1)       

        if controlkey == ord('q'):
            frame = detect_body(frame)
        if controlkey == ord('w'):
            frame = detect_face(frame)
        if controlkey == ord('e'):
            frame = detect_car(frame)
        if controlkey == ord('r'):
            frame = detect_license(frame)
        if controlkey == ord('t'):
            frame = detect_sign(frame)
        if controlkey == ord('y'):
            frame = detect_TrafficLight(frame)
        if controlkey == ord('a'):
            break
        
        cv2.imshow('detection', frame)
    Video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detection()