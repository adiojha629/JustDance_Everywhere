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
# import moveNet
from moveNet import outputFrame, lock, check,  get_web

print(check)


app = Flask(__name__)

outputFrame = None


# initialize the video stream 
# time.sleep(2.0)


# webpage to connect
@app.route('/')
def main():
    return render_template('main.html')

    # return "This is the homepage"

@app.route('/camera/')
def camera():
    # get_web()
    print ('camera got clicked!')
    return render_template('camera.html')
    # return render_template("video.html")

@app.route("/video_feed/")
def video_feed():
	# return the response generated along with the specific media
	# type (mime type)
  print("in video_feed")
  return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")
 

@app.route('/my-link/')
def my_link():
  print ('I got clicked!')
  return render_template("video.html")


def generate():
    # grab global references to the output frame and lock variables
    print("ADI!!!")
    t = threading.Thread(target=get_web(), args=())
    t.daemon = True
    print("Before thread")
    t.start()
    # loop over frames from the output stream
    print("After thread")
    adi = input("Enter")
    while True:
        # wait until the lock is acquired
        print("outside")
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')












if __name__ == '__main__':
  app.run(debug=True, threaded=True)
 



