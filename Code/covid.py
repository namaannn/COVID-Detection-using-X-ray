from flask import Flask,send_from_directory,redirect,url_for,request,render_template
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename
import os

app=Flask(__name__)
model=load_model('covid.h5')
def model_predict(filename,model):
    img=image.load_img(filename,target_size=(224,224))
    img=image.img_to_array(img)
    x=np.expand_dims(img,axis=0)
    pred=model.predict(x)
    preds=np.argmax(pred,axis=1)
    if preds==0:
        preds="Covid Positive"
    elif preds==1:
        preds="Normal"
    elif preds==2:
        preds="Viral Pneumonia"
    else:
        preds="Couldn't classify"
    return preds
@app.route('/',methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def upload():
    if request.method=='POST':
        f=request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath,'upload',secure_filename(f.filename))
        f.save(file_path)
        preds=model_predict(file_path,model)
        result=preds
        return result
    return None
if __name__=='__main__':
    app.run(debug=True)
        
