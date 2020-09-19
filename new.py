# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:55:55 2020

@author: Harsh Miglani
"""

from flask import Flask ,jsonify,request,json
from captionit import caption_this_image,predict_caption
app = Flask(__name__)
 
@app.route("/harsh",methods=['GET','POST'])
 
def tbh():
   if(request.method=='POST'):
       sj=request.files.get('image','')
       caption=caption_this_image(sj)
       return jsonify({"key": caption})
   return jsonify({"key":"end"})
 
   
app.run(host="localhost", port=int("7882"))