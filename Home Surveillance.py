import numpy as np
import argparse
import imutils
import time
import cv2
from imutils.video import VideoStream 
from imutils.video import FPS
from twilio.rest import Client 

#Set Up Twilio WhatsApp Client
account_sid = 'YOUR ACCOUNT SID HERE' 
auth_token = 'YOUR ACCOUNT TOKEN HERE' 
client = Client(account_sid, auth_token) 

ap = argparse.ArgumentParser()
ap.add_argument("-p","--prototxt",required = False, default='MobileNetSSD_deploy.prototxt.txt',help = 'path to Caffe prototxt file')
ap.add_argument("-m","--model", required = False, default = 'MobileNetSSD_deploy.caffemodel',help = 'pathto Caffe pre-trained model')
ap.add_argument("-c","--confidence",type=float, default=.2,help = 'minimum probability to fiter weak detections')
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair", "cow", "diningtable","dog", "horse", "motorbike", "person", "pottedplant", "sheep","sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt', 'MobileNetSSD_deploy.caffemodel')

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

msg_flag = True

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    (h,w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),0.007843,(300,300),127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > args["confidence"]:
            idx = int(detections[0,0,i,1])
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            
            if idx == 15:
                person_image = frame[startX:endX,startY:endX]
                gray = cv2.cvtColor(person_image, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray, 1.3, 5)
                #if msg_flag == True:
                for (a,b,c,d) in faces:
                    cv2.imwrite("detected.png", person_image[b:b+d,a:a+c])
                    message = client.messages.create( 
                            from_='whatsapp:+00000000000',  
                            body='INTRUDER DETECTED ! CHECK EMAIL FOR IMAGES',      
                            to='whatsapp:+00000000000')
                #msg_flag = False
                    
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence*100)
            cv2.rectangle(frame,(startX,startY),(endX,endY), COLORS[idx],2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame,label,(startX,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx],2)

    cv2.imshow("Surveillance System",frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    fps.update()

fps.stop()
#print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
#print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
cv2.destroyAllWindows()
vs.stop()