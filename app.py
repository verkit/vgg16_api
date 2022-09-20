import os
import numpy as np
import cv2
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
from keras.utils import img_to_array
from keras.models import load_model

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'temp'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
	return 'Hi, Whatsupp!'

@app.route('/predict', methods=['POST'])
def predict():
	# check if the post request has the file part
	if 'file' not in request.files:        
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	file = request.files['file']
	if file.filename == '':
		resp = jsonify({'message' : 'No file selected for uploading'})
		resp.status_code = 400
		return resp
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		file.save(path)
		pred = predict(path)		
		resp = jsonify({
			'message' : 'succesfully classify', 
			'value': pred,
			})
		resp.status_code = 201
		return resp
	else:
		resp = jsonify({'message' : 'Allowed file types are png, jpg, jpeg'})
		resp.status_code = 400
		return resp

file_model = 'vgg16_scratch.h5'
labels = ["pneumonia", "covid", "normal", "tbc"]

def predict(path):
	data = np.zeros((1, 224, 224, 3))
	img = cv2.imread(path)
	img = cv2.resize(img,(224,224))
	pil_img = Image.fromarray(img)
	data[0]+=img_to_array(pil_img)
	model = load_model(file_model)
	y_pred = model.predict(data)  

	dict_result = {}
	for i in range(4) :
		dict_result[y_pred[0][i]] = labels[i]		
	res = y_pred[0]
	res.sort()
	res = res[::-1]
	prob=res[:4]
	prob_result =[]
	class_result=[]
	for i in range(4) :
		prob_result.append ((prob[i]*100).round(2))
		class_result.append(dict_result[prob[i]])

	return class_result, prob_result