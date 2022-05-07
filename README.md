# JustDance_Everywhere
Spring 22 CV Project



### Play the Game

Under JustDance_Everywhere\[JSD_website](https://github.com/adiojha629/JustDance_Everywhere/tree/main/JSD_website) directory run command:

'python .\server.py' 

Note: currently the command have to run from the JSD_website directory.

After the website show up, click the button at the middle of the justdance video and the game would start in 10 seconds. 

### Change songs

Due to github size limit, currently there’s only one demo dance. If you want to try other dance, download the video from the following google drive https://drive.google.com/drive/folders/1tzGKYIqvO4bdjpQxuxs3x8DDdH4sMVVp?usp=sharing .

Place the <dance>.mp4 in \JustDance_Everywhere\JSD_website\static\

Change the dance video in \JustDance_Everywhere\JSD_website\camera.html to the dance you want to play.

' Line 36: <source src="{{ url_for('static', filename='justdance_happy_short.mp4') }}" '

Also change the angle data in \JustDance_Everywhere\JSD_website\moveNet.py to the corresponding angle.csv 

' Example: Line 254: justDance_data = pd.read_csv( 'Angles CSV2/ JustDance_Levitating_angle.csv') '

Then run the python .\server.py to start your New dance! 



### JSD_website





​	







