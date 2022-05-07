import tensorflow as tf
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import threading
import cv2
import time
from imutils.video import VideoStream


global outputFrame
global lock
outputFrame = None
lock = threading.Lock()

interpreter = tf.lite.Interpreter(model_path='.\lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()


joint_to_index_moveNet = {
      'left_shoulder': [7, 5,6],
      'right_shoulder': [8, 6, 5],
      'left_elbow': [5, 7, 9],
      'right_elbow': [6, 8, 10],
      'left_knee': [11, 13, 15],
      'right_knee': [12, 14, 16]
}


EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}



def get_mse(angle_person, angle_Jdance, body_part = 'upper_body', 
            joint_names = {
                'upper_body':['left_shoulder','right_shoulder','left_elbow','right_elbow'],#,'left_neck','right_neck'],
                'lower_body':['left_knee','right_knee']
                    }): 
  '''Get MSE for upper/lower body
  params:
  angle_person : feature_vector
  angle_Jdance : feature_vector
  body_part : 'upper_body' or 'lower_body', indicate which part you want to compute mse for 
  return mse 

  Sample code:
   f1 = open('person.json')
   data1 = json.load(f1)
   f2 = open('JustDance.json')
   data2 = json.load(f2)
   angle_person = get_angles(data1)
   angle_Jdance = get_angles(data2)
   get_mse(angle_person,angle_person)
    will return MSE for upper body

  get_mse(angle_person, angle_person, body_part = 'lower_body')
    will return MSE for lower body'''
  sum = 0
  num_of_valid_angle = 0
  num_angle = 0
  penalty = 1/10* 180**2
  for joint_name in joint_names[body_part]:
      _,an_angle_person = angle_person[joint_name]
      _,an_angle_Jdance = angle_Jdance[joint_name]
      if an_angle_person != 9999 and an_angle_Jdance != 9999 : # If either of the angle is invalid, we will not compute mse
          num_of_valid_angle +=1
          sum += (an_angle_person-an_angle_Jdance)**2
      else:
          sum += penalty
      num_angle += 1
  if num_of_valid_angle ==0: # if no angle valid, we will return infinity
      return np.Infinity
  # print("End")
  return sum/num_angle


def draw_keypoints(frame, keypoints, confidence):
  """
  Draw keypoints onto the frame.
  params:
  frame: 3D numpy array
  keypoints: 3D numpy array (y,x,confidence)
  confidence: threashold for confidence
  """
  y, x, c = frame.shape
  shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
  for kp in shaped:
      ky, kx, conf = kp
      if conf > confidence:
          cv2.circle(frame,(int(kx),int(ky)), 4,(0,255,0),-1)


def get_angles_moveNet(data):
    '''Get feature vectors in the format like:
              {'left_elbow': [3, 152],
            'left_knee': [10, 160],
            'left_neck': [1, 112],
            'left_shoulder': [2, 108],
            'right_elbow': [6, 166],
            'right_knee': [13, 167],
            'right_neck': [1, 64],  //not supported 
            'right_shoulder': [5, 149]}
    param:
    data:  2D array 
      directly from the moveNet output 
    return:
      the feature vectors (dictionay) with angles, the angle will be 9999 if one
      of the joints is invalid (i.e. the confidence is 0)
    '''
    coords_np = data  # shape (17,3)
    feature_vector = {}

  # list_of_joints = [left_shoulder,right_shoulder,left_elbow,right_elbow,left_knee,right_knee, left_neck, right_neck]
    for joint in joint_to_index_moveNet.keys():
        # index = [i,j,k] where i,j,k refer to body17 's numbering of joints
        # coords_np[i] - gets us the x,y coordinate of body17 joint i
        index = joint_to_index_moveNet[joint]
        pt1, pt2, pt3 = coords_np[index[0]
                                  ], coords_np[index[1]], coords_np[index[2]]
        confidence_1 = coords_np[index[0]][2]
        confidence_2 = coords_np[index[1]][2]
        confidence_3 = coords_np[index[2]][2]
        angle = 0
        for c in [confidence_1, confidence_2, confidence_3]:
            if c <= 0.02:
                print("Angle Won't make Sense")
                angle = 9999  # we'll check for this value when calculating mse
                break
        if angle!=9999:
            angle = find_angle(pt1, pt2, pt3)
        feature_vector[joint] = [index[1], int(angle)]
    assert len(feature_vector) == 6
    return feature_vector

def find_angle(pt1, pt2, pt3):
    """
    Find the angle between three points.
    inspiration: https://stackoverflow.com/questions/2049582/how-to-determine-the-angle-between-3-points
    """
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)
    theta = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(theta * 180/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle

def draw_connections(frame, keypoints, edges, confidence_threshold):
  """
  Draw connections between keypoints onto the frame.
  params:
  frame: 3D numpy array
  keypoints: 3D numpy array (y,x,confidence)
  edges: list of tuples (joint_name1, joint_name2)
  confidence_threshold: threashold for confidence
  """
  y, x, c = frame.shape
  shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
  for edge, color in edges.items():
      p1, p2 = edge
      y1, x1, c1 = shaped[p1]
      y2, x2, c2 = shaped[p2]
      if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
          cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


def get_score(angle_person, angle_Jdance, body_part = 'upper_body',
            joint_names = {
                'upper_body':['left_shoulder','right_shoulder','left_elbow','right_elbow'],
                'lower_body':['left_knee','right_knee']},
              thresholds = [0.9, 0.8, 0.6, 0.3, 0.15],score_rate = 0.0005):  #or 0.9 for perfect
    """
    Get the score for a given body part for specific frame.
    params:
    angle_person: dictionary of angles from person
    angle_Jdance: dictionary of angles from Just Dance video
    body_part: string, either 'upper_body' or 'lower_body'
    return:
    score: float, between 0 and 1
    score_str: string, the score represented as perfect,super, good, nice, ok, and x
    """
    mse = get_mse(angle_person, angle_Jdance, body_part,
            joint_names)
    score = 1- np.tanh(score_rate*mse)
    # score = 1 - mse/(45**2)
    # 0.9 Perfect, 0.8 Super, 0.6 Good, 0.4 Nice, 0.1 Ok, 0 X
    assert(len(thresholds) == 5)
    if score >= thresholds[0]:
      return score, "PERFECT! "
    elif score >= thresholds[1]:
      return score, "SUPER! "
    elif score >= thresholds[2]:
      return score, "GOOD! "
    elif score >= thresholds[3]:
      return score, "NICE! "
    elif score >= thresholds[4]:
      return score, "OK "
    else:
      return score, "X"

def get_final_score(scores,
              thresholds = [0.9, 0.8, 0.6, 0.4, 0.1]):
    """
    Get the final score for the game
    """
    final_score = np.mean(scores)
    assert(len(thresholds) == 5)
    if final_score >= thresholds[0]:
      return  "SSS "
    elif final_score >= thresholds[1]:
      return " SS "
    elif final_score >= thresholds[2]:
      return " S "
    elif final_score >= thresholds[3]:
      return  "A "
    elif final_score >= thresholds[4]:
      return " B "
    else:
      return  "C "

def get_web():
  """
  Run MoveNet
  Calculate the angles and loss(MSE)
  update the outputFrame 
  """
  global outputFrame, lock
  # Setting frame rate idea from here:
  # https://www.programcreek.com/python/example/114226/imutils.video.VideoStream
  frame_p_sec = 30
  frame_idx = 0 
  vs = VideoStream(src=0,framerate=frame_p_sec).start()
  # the openpose angle data. Change the source is you want to play another dance
  justDance_data = pd.read_csv(
      'Angles CSV/angles.csv')
  justDance_data = justDance_data.drop(['Unnamed: 0'], axis=1)
  num_points = len(justDance_data) # number of "frame_idx//frame_p_sec"'s we have
  scores = []
  start_time = time.time()
  start = False
  json_index = 1
  last_time = 0

  # Inerate through all the angles in dance data file
  # Compare the angles with the dancer's angles
  while json_index < num_points:
      frame = vs.read()   
      frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
      frame = cv2.flip(frame, 1)
      if len(scores):# display score if we have one
          frame = cv2.putText(frame,str(scores[-1]),(350,450),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                          fontScale=2,color=(0,255,0),thickness=4,lineType=cv2.LINE_AA)
      if not start and time.time() - start_time >= 10:
          start = True

      # Run the Movenet Once every second
      # time units are seconds
      if time.time() - last_time >= 0.99 and start: 
          last_time = time.time()
          img = frame.copy()
          img = np.squeeze(img)

         
          img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
          # for thunder resize to 256
          # img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
          
          input_image = tf.cast(img, dtype=tf.float32)
          # setup input and output
          input_details = interpreter.get_input_details()
          output_details = interpreter.get_output_details()

          # make predictions
          interpreter.set_tensor(
              input_details[0]['index'], np.array(input_image))
          interpreter.invoke()
          keypoints_with_scores = interpreter.get_tensor(
              output_details[0]["index"])
          keypoints = np.squeeze(keypoints_with_scores)

          # draw keypoints and skeleton
          draw_connections(frame, keypoints_with_scores, EDGES, 0.2)
          draw_keypoints(frame, keypoints_with_scores, 0.2)

          # get the angles in dictionary format
          webcam_angle = get_angles_moveNet(keypoints)
          jd_frame = justDance_data.iloc[json_index].to_dict()

          # get the score for the frame
          for key in jd_frame.keys():# [2,3]
              jd_frame[key] = eval(jd_frame[key])
          score, score_str = get_score(angle_person=webcam_angle, angle_Jdance=jd_frame)
          scores.append(score_str)
          json_index+=1
      with lock:
          outputFrame = frame.copy()    
          
  print("The end")
  vs.stop()


if __name__ == '__main__':
    get_web()

