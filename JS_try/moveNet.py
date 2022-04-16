import tensorflow as tf
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import cv2
import time

interpreter = tf.lite.Interpreter(model_path='lite-model_movenet_singlepose_lightning_3.tflite')
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
  >>> f1 = open('person.json')
  >>> data1 = json.load(f1)
  >>> f2 = open('JustDance.json')
  >>> data2 = json.load(f2)
  >>> angle_person = get_angles(data1)
  >>> angle_Jdance = get_angles(data2)
  >>> get_mse(angle_person,angle_person)
    will return MSE for upper body

  >>>get_mse(angle_person, angle_person, body_part = 'lower_body')
    will return MSE for lower body

  '''

  sum = 0
  num_of_valid_angle = 0

  for joint_name in joint_names[body_part]:
    _,angle_person = angle_person[joint_name]
    _,angle_Jdance = angle_Jdance[joint_name]

    if angle_person != 9999 and angle_Jdance != 9999 : # If either of the angle is invalid, we will not compute mse
        num_of_valid_angle +=1
        sum += (angle_person-angle_Jdance)**2

    if num_of_valid_angle ==0: # if no angle valid, we will return infinity
      return np.Infinity

    return sum/num_of_valid_angle


def draw_keypoints(frame, keypoints, confidence):
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
      the feature vectors with angles, the angle will be 9999 if one
      of the joints is invalid (i.e. the confidence is 0)

  '''
  # preprocessing
    coords_np = data  # shape (17,3)
#   coords_np = np.array(coords).reshape(25,3) #Group by coordinates [x,y,confidence]

    feature_vector = {}

  # Define joints index for body25

  # joint_to_index['left_shoulder'] = [1,2,3]
  # right_shoulder = [1,5,6]
  # left_elbow = [2,3,4]
  # right_elbow = [5,6,7]
  # left_knee = [9,10,11]
  # right_knee = [12,13,14]
  # left_neck = [0,1,2]
  # right_neck = [0,1,5]

  # list_of_joints = [left_shoulder,right_shoulder,left_elbow,right_elbow,left_knee,right_knee, left_neck, right_neck]

    for joint in joint_to_index_moveNet.keys():
        # index = [i,j,k] where i,j,k refer to body 25's numbering of joints
        # coords_np[i] - gets us the x,y coordinate of body 25 joint i
        index = joint_to_index_moveNet[joint]
        pt1, pt2, pt3 = coords_np[index[0]
                                  ], coords_np[index[1]], coords_np[index[2]]
        confidence_1 = coords_np[index[0]][2]
        confidence_2 = coords_np[index[1]][2]
        confidence_3 = coords_np[index[2]][2]
        if 0 in [confidence_1, confidence_2, confidence_3]:
            print("Angle Won't make Sense")
            angle = 9999  # we'll check for this value when calculating mse
        else:
            angle = find_angle(pt1, pt2, pt3)

        # angle = find_angle(pt1,pt2,pt3)
        feature_vector[joint] = [index[1], int(angle)]
  # feature_vector = np.array(feature_vector)
    assert len(feature_vector) == 6
  #print(feature_vector)
    return feature_vector

def find_angle(pt1, pt2, pt3):
    #     y, x, c = frame.shape
    #     shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    a = np.array(pt1)
    b = np.array(pt2)
    c = np.array(pt3)
    theta = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(theta * 180/np.pi)
    if angle > 180:
        angle = 360 - angle
    return angle


def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]
        if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)


def get_web():

    # sleep and count down  
    time.sleep(1)
    for i in range(20):
        print(20-i)
        time.sleep(1)

    cap = cv2.VideoCapture(0)
    frame_idx = 0 
    happy_short_data = pd.read_csv(
        'Angles CSV/angles.csv')
    happy_short_data = happy_short_data.drop(['Unnamed: 0'], axis=1)
    print(happy_short_data.head())
    while cap.isOpened():
        ret, frame = cap.read()  # fram- image
        if ret:
            frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
            cv2.imshow('MoveNet Lightning', frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

            # check if we should run the network
            if frame_idx % 30 == 0:
                # do network stuff
                #input: A frame of video or an image, represented as an
                #float32 tensor of shape: 192x192x3. Channels order: RGB with values in [0, 255].
                #reshape
                img = frame.copy()
                # for lignting resize to 192
                # img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
                img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
                
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



                # print(keypoints_with_scores)
                # draw_connections(frame, keypoints_with_scores, EDGES, 0.2)
                # draw_keypoints(frame, keypoints_with_scores, 0.2)
                # get the angles in dictionary format
                webcam_angle = get_angles_moveNet(keypoints)
                jd_frame = happy_short_data.iloc[frame_idx//30].to_dict()
                for key in jd_frame.keys():# [2,3]
                    jd_frame[key] = eval(jd_frame[key])
                score = get_mse(angle_person=webcam_angle, angle_Jdance=jd_frame, body_part = 'upper_body', 
                    joint_names = {
                        'upper_body':['left_shoulder','right_shoulder','left_elbow','right_elbow'],#,'left_neck','right_neck'],
                        'lower_body':['left_knee','right_knee']
                            })
                print(frame_idx," ",score)
                # scores.append(score)


                # angles.append(get_angles_moveNet(keypoints))
            frame_idx+=1
        
    # cap.release()
    # cv2.destroyAllWindows()



# def get_web():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         cv2.imshow('MoveNet Lightning',frame)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break


# get_web()

