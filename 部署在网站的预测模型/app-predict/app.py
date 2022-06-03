from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from dataloader.DataLoader import pred_data_variable
from dataloader.DataLoader import Instance
from TaggerModel_easy import POSTagger
import os
import re
# TensorFlow and tf.keras
import torch
import pickle
# Some utilites
import numpy as np

#Declare a flask app
app = Flask(__name__)

def load_model(model_path):
    """

    Args:
        model_path : 模型放置路径

    Returns:
    模型文件
    """
    assert os.path.exists(model_path) and os.path.isfile(model_path)
    # GPU上训练的模型在CPU上运行
    model = torch.load(model_path, map_location='cpu')  # Load all tensors onto the CPU
    return model

print('Model loaded. Start serving...')

def load_vocab(vocab_path):
    assert os.path.exists(vocab_path) and os.path.isfile(vocab_path)
    with open(vocab_path, 'rb') as fin:
        vocab = pickle.load(fin)
    return vocab

def predict(pred_data):
    tagger = load_model('models/pos_model.pkl')
    tagger.eval()
    vocab = load_vocab('models/vocab.pkl')
    char_vocab = load_vocab('models/char_vocab.pkl')
    xb, xch, seq_lens = pred_data_variable(pred_data, vocab, char_vocab)
    pred = tagger(xb, xch, seq_lens)  # batch_size * tag_size
    pred_ids = torch.argmax(pred, dim=1)  # (batch_size, )
    return vocab.index2pos(pred_ids.tolist())

@app.route('/', methods=['GET'])
def index():
    #Main Page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def model_predict():
    if request.method == 'POST':
    
        pred_data = request.form['message']
        
        #Get the image from post request
        pred_data = pred_data.split()
        
        #save the image to ./uploads
        #img.save("./uploads/image.png")
        data = [Instance(pred_data, None)]
        #Make prediction
        pos = predict(data)
        
        #process your result for human
        dict = {"VA":"谓词性形容词",	"VC":"系动词",	"VE":"主要动词",	"VV":"其他动词",	"VV-2":"其他动词",	"NR":"专有名词",	"NR-SHORT":"专有名词",	"NT":"时间名词",	"NT-SHORT":"时间名词",	"NN":"其他名词",	"NN-SHORT":"其他名词",	"LC":"方位词",	"PN":"指代",	"DT":"限定词",	"CD":"基数词",	"OD":"序列词",	"M":"度量衡",	"AD":"副词",	"P":"介词",	"CC":"并列",	"CS":"从属",	"DEC":"补语/名词化标记",	"DEG":"关联/所有格标记",	"DER":"补语短语",	"DEV":"方式",	"SP":"句末助词",	"AS":"动态助词",	"AS-1":"动态助词",	"ETC":"等等",	"MSP":"其他助词",	"MSP-2":"其他助词",	"IJ":"感叹词",	"ON":"拟声词",	"PU":"标点符号",	"JJ":"其他名词修饰语",	"FW":"外来词",	"LB":"长被结构",	"SB":"短被结构",	"BA":"把字结构",	"URL":"网址",	"NOI":"",		"X":""}
        list1 = [dict[po] for po in pos]
        
        return render_template('index.html', prediction = str(list1))
    
if __name__ == '__main__':
    http_server = WSGIServer(('127.0.0.1', 5000), app)
    http_server.serve_forever()
