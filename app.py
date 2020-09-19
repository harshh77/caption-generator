# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 17:52:10 2020

@author: Harsh Miglani
"""

from flask import Flask, render_template, url_for, request, redirect
from captionit import caption_this_image
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/predict')
def ups():
	img = request.args.get('image')
	
	caption = caption_this_image(img)
		
	result_dic = {
			'description' : caption
		}
	return result_dic

app.run(host="localhost", port=int("777"))

# @app.route('/predict')
# def ups():
# 	img = request.args.get('image',' ')
# 	
# 	caption = caption_this_image(img)
# 		
# 	result_dic = {
# 			'description' : caption
# 		}
# 	return result_dic

# app.run(host="localhost", port=int("777"))


if __name__ == '__main__':
	app.run(debug = True)
    
# @app.route('/abc', methods=['GET'])
# def just():
#     if request.method == 'GET':
#         result={'description':"HELOZ"}
#     return result

# app.run(host="localhost", port=int("777"))
        
# @app.route('/')
# def hello():
#     return render_template('index.html')

# def index():
#     return "Welcome to Python Server"


		# print(img)
		# print(img.filename)

		#img.save("static/"+img.filename)