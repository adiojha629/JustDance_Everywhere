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
import moveNet
# from moveNet import outputFrame, lock, vs, check,  get_web

print(moveNet.check)


app = Flask(__name__)

outputFrame = None
lock = threading.Lock()


# initialize the video stream 
# time.sleep(2.0)


# webpage to connect
@app.route('/')
def main():
    return render_template('main.html')

    # return "This is the homepage"

@app.route('/camera/')
def camera():
    moveNet.get_web()
    return ""
    # return render_template("video.html")
 

@app.route('/my-link/')
def my_link():
  print ('I got clicked!')
  return render_template("video.html")



if __name__ == '__main__':
  app.run(debug=True)
