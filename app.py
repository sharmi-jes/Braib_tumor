from flask import Flask,request,render_template,send_from_directory
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from PIL import Image


# load the model
model=load_model('E:\DEEP_LEARNING_PROJECTS\Brain_Tumor_Project\model(3).h5')

# app created 
app=Flask(__name__)


# create dynamically upload floder
UPLOAD_FOLDER='./uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# clas labels
class_labels = sorted(os.listdir(r'E:\DEEP_LEARNING_PROJECTS\Brain_Tumor_Project\train'))

# define predict_tumor function
def predict_tumor(img,img_size=128):
    # load the image
    img=load_img(img,target_size=(img_size,img_size))

    # convert the img to numpy array
    img=img_to_array(img)/255

# add expand dims to the array
    img=np.expand_dims(img,axis=0)
    

    # predict the model
    prediction=model.predict(img)

    # apply argmax function for probabailities
    prediction_class_index=np.argmax(prediction,axis=1)[0]


    # appply max function to get max value
    confidence=np.max(prediction,axis=1)[0]

    if class_labels[prediction_class_index]=="Notumor":
        return f"NoTumor",confidence
    else:
        return f"Tumor:{class_labels[prediction_class_index]}",confidence
    


# define routes
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict",methods=["GET","POST"])
def index():
    if request.method=="POST":
        # file handle
        file=request.files["file"]

        if file:
            file_location=os.path.join(app.config["UPLOAD_FOLDER"],file.filename)
            file.save(file_location)

            result,confidence=predict_tumor(file_location)

            # Return result along with image path for display
            return render_template('index.html', result=result, confidence=f"{confidence*100:.2f}%", file_path=f'/uploads/{file.filename}')

    return render_template('index.html', result=None)

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)

