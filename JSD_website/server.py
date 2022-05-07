# reference:https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/ 
from imutils.video import VideoStream
from flask import Response
from flask import Flask, render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
from moveNet import lock, get_web
import moveNet



app = Flask(__name__)

# webpage to connect
@app.route('/')
def main():
    print ('camera got clicked!')
    return render_template('camera.html')
   

@app.route("/video_feed/")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
  print("in video_feed")
  return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

def generate():
    """
    grab global references to the output frame and lock variables
    iterate through frames from the output stream
    """
    #start a thread to run moveNet
    t = threading.Thread(target=get_web)
    t.daemon = True
    t.start()
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            if moveNet.outputFrame is None:
                print("None")
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", moveNet.outputFrame)
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')

if __name__ == '__main__':
  app.run(debug=True, threaded=True)