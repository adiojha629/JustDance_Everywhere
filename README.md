# JustDance_Everywhere
Spring 22 CV Project

![hippo](Others/JustDance_giif.gif)

### Play the Game

Under JustDance_Everywhere/[JSD_website](https://github.com/adiojha629/JustDance_Everywhere/tree/main/JSD_website) directory run command:

'python .\server.py' 

Note: currently the command have to run from the JSD_website directory.

After the website show up, click the button at the middle of the justdance video and the game would start in 10 seconds. 

### Change songs

Due to github size limit, currently there’s only one demo dance. If you want to try other dance, download the video from the following google drive https://drive.google.com/drive/folders/1tzGKYIqvO4bdjpQxuxs3x8DDdH4sMVVp?usp=sharing .

Place the  dance.mp4 in  \JustDance_Everywhere\JSD_website\static\ .Change the dance video in \JustDance_Everywhere\JSD_website\camera.html to the dance you want to play.
```python
Line 36: <source src="{{ url_for('static', filename='justdance_happy_short.mp4') }}" 
```
Also change the angle data in \JustDance_Everywhere\JSD_website\moveNet.py to the corresponding angle.csv. Then run the python .\server.py to start your New dance! 

Example:
```python
Line 254: justDance_data = pd.read_csv( 'Angles CSV2/ JustDance_Levitating_angle.csv') 
```


### [JSD_website](https://github.com/adiojha629/JustDance_Everywhere/tree/main/JSD_website)

This directory includes all the codes and the datasource for the website.

    JSD_website/static: store the movenet model, dance video and the css file.

    JSD_website/templates: the html page for the website.

    JSD_website/Angle_CSV(2): the angle data of the JustDance video (Captured using OpenPose).

    JSD_website/MoveNet.py: The main backend code file for the website. Run MoveNet on the streaming video and sent scores.

    JSD_website/server.py: Flask backend 

### [Code_run_on_colab](https://github.com/adiojha629/JustDance_Everywhere/tree/main/code_run_on_colab)

The Google colab file contains codes that we used to :

1. Download the video from YouTube.
2. Sample the video to 1 frame/sec.
3. Do pose estimation to the sampled video.
4. Get joint angles.
5. Scoring based on those angles using our scoring function.

In detail, it contains:

1. Preprocessing for JustDance Video

     1.1. Install OpenPose

     1.2. Download video from YouTube

     1.3. Sample the Video

     1.4. Run Openpose, detect poses, stored the keypoints a json files

2.  Scoring Functions

      2.1. Read json file (Generated by OpenPose) from JustDance Video

      2.2. Get joins angles

      2.3. Define the function to find angle from OpenPose

      2.4. Define the function to find angle from MoveNet

      2.5. Functions for pose estimation visualization

      2.6. Compute MSE between angles

      2.7. Threshold score to "Perfect, Good, Okay,..." use our score function

      2.8. Plot diagram to compare our score function and NMSE function

   

   

   ### [Others](https://github.com/adiojha629/JustDance_Everywhere/tree/main/Others)

   These programs were made during the developing stage of the project and are no longer used for the final result. This includes code used to run other pose estimation models and different versions of the game interface.
   
   ### Coding Reference
   
   [opencv stream video to web browser](https://pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/)
   
   [MoveNetLightning](https://github.com/nicknochnack/MoveNetLightning)
   
   [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
   
   [CSS play button](https://codepen.io/Griezzi/pen/mdOzrWP)
   
   [Find Angle](https://stackoverflow.com/questions/2049582/how-to-determine-the-angle-between-3-points)
   
   [youtube downloader](https://towardsdatascience.com/build-a-youtube-downloader-with-python-8ef2e6915d97)
   

   
   
   
   
