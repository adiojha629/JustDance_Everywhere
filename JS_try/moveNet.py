# from tkinter.messagebox import NO
import tensorflow as tf
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
import threading
import cv2
import time
from imutils.video import VideoStream


# vs, outputFrame, lock
global outputFrame
global lock
outputFrame = None
lock = threading.Lock()
check = 1
# temp2 = input("press any key to continue")
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
#   '''Get MSE for upper/lower body
#   params:
#   angle_person : feature_vector
#   angle_Jdance : feature_vector
#   body_part : 'upper_body' or 'lower_body', indicate which part you want to compute mse for 

#   return mse 

#   Sample code:
#    f1 = open('person.json')
#    data1 = json.load(f1)
#    f2 = open('JustDance.json')
#    data2 = json.load(f2)
#    angle_person = get_angles(data1)
#    angle_Jdance = get_angles(data2)
#    get_mse(angle_person,angle_person)
#     will return MSE for upper body

#   get_mse(angle_person, angle_person, body_part = 'lower_body')
#     will return MSE for lower body'''
    sum = 0
    num_of_valid_angle = 0
    penalty = 1/10* 180**2
    # print("Beginning")
    for joint_name in joint_names[body_part]:
    # joint_name = joint_names[body_part][0]
        # print("Before")
        # print(angle_person)
        _,an_angle_person = angle_person[joint_name]
        # print("After")
        # print(an_angle_person)
        _,an_angle_Jdance = angle_Jdance[joint_name]

        if an_angle_person != 9999 and an_angle_Jdance != 9999 : # If either of the angle is invalid, we will not compute mse
            num_of_valid_angle +=1
            sum += (an_angle_person-an_angle_Jdance)**2
        else:
            sum += penalty

    if num_of_valid_angle ==0: # if no angle valid, we will return infinity
        return np.Infinity
    # print("End")
    return sum/4


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
        angle = 0
        for c in [confidence_1, confidence_2, confidence_3]:
            if c <= 0.02:
                print("Angle Won't make Sense")
                angle = 9999  # we'll check for this value when calculating mse
                break
        if angle!=9999:
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


def Count_Down(num_seconds):
    '''
    A count down before the camera and the video start!
    @param: num_seconds (int): number of seconds to count down

    I use open-cv's window commands to display the count down.
    Note: don't use time.sleep for waiting; it prevents open cv from displaying windows
    We use waitKey instead

    Code inspired by responses here:
    https://stackoverflow.com/questions/31350240/python-opencv-open-window-on-top-of-other-applications
    https://www.geeksforgeeks.org/python-opencv-cv2-puttext-method/
    '''
    cv2.namedWindow('CountDown!', cv2.WINDOW_NORMAL)
    img = np.zeros((400,400,3))
    for i in range(num_seconds):
        print(num_seconds-i)
        time_left = cv2.putText(img.copy(),str(num_seconds-i),(200,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
        # use img.copy() so we don't write numbers on top of one another
        cv2.imshow('CountDown!',time_left)
        cv2.waitKey(1000)
    print("Start!")
    start_text = cv2.putText(img.copy(),"Start!",(200,200),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=1,color=(255,255,255),thickness=2,lineType=cv2.LINE_AA)
    cv2.imshow("CountDown!",start_text)
    cv2.waitKey(1000)
    cv2.destroyWindow("CountDown!")




def get_score(angle_person, angle_Jdance, body_part = 'upper_body',
            joint_names = {
                'upper_body':['left_shoulder','right_shoulder','left_elbow','right_elbow'],
                'lower_body':['left_knee','right_knee']},
              thresholds = [0.85, 0.8, 0.6, 0.3, 0.15],score_rate = 0.0005):  #or 0.9 for perfect
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
      return  " C "

def get_web():
    # global 
    # sleep and count down  
    # cap = cv2.VideoCapture(0)
    global outputFrame, lock
    frame_p_sec = 30
    frame_idx = 0 
    # Setting frame rate idea from here:
    # https://www.programcreek.com/python/example/114226/imutils.video.VideoStream
    vs = VideoStream(src=0,framerate=frame_p_sec).start()
    happy_short_data = pd.read_csv(
        'Angles CSV/angles.csv')
    happy_short_data = happy_short_data.drop(['Unnamed: 0'], axis=1)
    # print(happy_short_data.head())
    num_points = len(happy_short_data) # number of "frame_idx//frame_p_sec"'s we have
    scores = []
    start_time = time.time()
    start = False
    json_index = 1

    # assert num_points == 60
    print("Num Points \n\n")
    print(num_points)
    last_time = 0
    # while frame_idx//30 < num_points:
    # while not start:
    #     frame = vs.read()
    #     with lock:
    #         # print("I have the lock")
    #         # print(frame)
    #         outputFrame = frame.copy()
    #     if not start and time.time() - start_time >= 5:
    #         start = True

    # adi = input('kevin')
    while json_index < num_points:
        # time.sleep(60.0 - ((time.time() - start_time) % 60.0))
        frame = vs.read()   
        # print("asdf",frame.shape)
        # ret, frame = cap.read()  # fram- image
        # if ret:  modification ??
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        frame = cv2.flip(frame, 1)
        if len(scores):# display score if we have one
            frame = cv2.putText(frame,str(scores[-1]),(350,450),fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=2,color=(0,255,0),thickness=4,lineType=cv2.LINE_AA)
        # cv2.imshow('MoveNet Lightning', frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        if not start and time.time() - start_time >= 10:
            start = True
            # print("Start!")

        # check if we should run the network
        # if frame_idx % 30 == 0 and start:
        
        if time.time() - last_time >= 0.99 and start: # units are seconds
            last_time = time.time()
            print("frame_idx:", frame_idx)
            print("time diff:", time.time()- start_time)
            # start_time = time.time()
            # do network stuff
            #input: A frame of video or an image, represented as an
            #float32 tensor of shape: 192x192x3. Channels order: RGB with values in [0, 255].
            #reshape
            img = frame.copy()
            img = np.squeeze(img)
            
            # for lignting resize to 192
            # img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 256, 256)
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            # print("kjlkasdf", img.shape)
            
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
            draw_connections(frame, keypoints_with_scores, EDGES, 0.2)
            draw_keypoints(frame, keypoints_with_scores, 0.2)
            # get the angles in dictionary format
            webcam_angle = get_angles_moveNet(keypoints)
            jd_frame = happy_short_data.iloc[json_index].to_dict()
            for key in jd_frame.keys():# [2,3]
                jd_frame[key] = eval(jd_frame[key])
            # score = get_mse(angle_person=webcam_angle, angle_Jdance=jd_frame)
            score, score_str = get_score(angle_person=webcam_angle, angle_Jdance=jd_frame)
            # score = 1 - np.tanh(0.001*score)
            # print(frame_idx," ",score)
            scores.append(score_str)
            json_index+=1
            # temp = input("Press any key to continue...")
        
        # frame_idx+=1
        # cv2.imshow('MoveNet Lightning', frame)
        with lock:
            # print("I have the lock")
            # print(frame)
            outputFrame = frame.copy()
        
            
    print("The end")
    vs.stop()
    # scores = np.array(scores)
    # print(scores)
    # scores = 1 - np.tanh(0.001*scores)
    # print(scores)
    # plt.hist(scores)
    # plt.show()

    
        
    # cap.release()
    # cv2.destroyAllWindows()



# countdown for 10 seconds

# def get_web():
#     cap = cv2.VideoCapture(0)
#     while cap.isOpened():
#         ret, frame = cap.read()
#         cv2.imshow('MoveNet Lightning',frame)
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break


if __name__ == '__main__':
    get_web()
    # Count_Down(10)

