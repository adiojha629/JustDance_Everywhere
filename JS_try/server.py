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
from moveNet import lock, check, get_web
import moveNet
# import moveNet
print(check)
t = 0


app = Flask(__name__)

# moveNet.outputFrame 



# initialize the video stream 
# time.sleep(2.0)


# webpage to connect
@app.route('/')
def main():
    print ('camera got clicked!')
    return render_template('camera.html')
    # return render_template('main.html')

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


@app.route("/video_only/")
def video_only():
	# return the response generated along with the specific media
	# type (mime type)
  print("in video_only")
  return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

 

@app.route('/my-link/')
def my_link():
  print ('I got clicked!')
  return render_template("video.html")


def generate():
    # grab global references to the output frame and lock variables
    # print("ADI!!!")
    # loop over frames from the output stream
    t = threading.Thread(target=get_web)
    t.daemon = True
    # print("After thread")
    t.start()
    # adi = input("Enter")
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if moveNet.outputFrame is None:
                print("None")
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", moveNet.outputFrame)
            
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        # cv2.imshow("Frame", outputFrame)
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
              bytearray(encodedImage) + b'\r\n')





if __name__ == '__main__':
  app.run(debug=True, threaded=True)
  # t = threading.Thread(target=get_web(), args=(),)
  # t.daemon = True
  

  
  
 



