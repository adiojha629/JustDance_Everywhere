from flask import Flask, render_template
from moveNet import get_web
app = Flask(__name__)

# webpage to connect
@app.route('/')
def main():
    return render_template('main.html')

    # return "This is the homepage"

@app.route('/camera/')
def camera():
    get_web()
    # return render_template("video.html")
 

@app.route('/my-link/')
def my_link():
  print ('I got clicked!')
  return render_template("video.html")



if __name__ == '__main__':
  app.run(debug=True)