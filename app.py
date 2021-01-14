# -*- coding: utf-8 -*-
import os
import numpy as np
import urllib.request
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
from PIL import Image
import cv2
import io

from keras import models
from keras.models import load_model
# 初回はVGG16 訓練済みモデル(540MB)をダウンロードするために50分ほど時間がかかる
# 訓練済みモデルの保存場所　カレントディレクトリの中の　/.keras　に作られる．
# /.keras/model/vgg16_weights_tf_dim_ordering_tf_kernels.h5
#from keras.applications.vgg16 import VGG16
#from keras.applications.vgg16 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
#import keras.preprocessing.image as Image

from keras.applications.mobilenet import MobileNet, preprocess_input, decode_predictions
from keras.layers import Input

app = Flask(__name__)

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'PNG', 'JPG'])
IMAGE_WIDTH = 640
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send', methods=['GET', 'POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']

        # 変なファイル弾き
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
        else:
            return ''' <p>許可されていない拡張子です</p> '''

        # BytesIOで読み込んでOpenCVで扱える型にする
        f = img_file.stream.read()
        bin_data = io.BytesIO(f)
        file_bytes = np.asarray(bytearray(bin_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # とりあえずサイズは小さくする
        img = cv2.resize(img, (640, 480))
        raw_img = cv2.resize(img, (320, 240))

        # サイズだけ変えたものも保存する
        raw_img_url = os.path.join(app.config['UPLOAD_FOLDER'], 'raw_' + filename)
        cv2.imwrite(raw_img_url, raw_img)

        # なにがしかの加工
        gray_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)

        # 「imagenet 」の入力データを指定
        image_rows = 224  # 画像縦の画素数
        image_cols = 224  # 画像横の画素数
        image_color = 3  # 画素の色数3/R[0,255] G[0,255] B[0,255]]
        image_shape = (image_rows, image_cols, image_color)  # 画像のデータ形式
        image_size = image_rows * image_cols * image_color  # 画像のニューロン数150528

        # 訓練した学習器とパラメータの読み込み(CNN-VGG16 訓練済みモデル　540MB)
        #model_path = "./weights/resnet50_weights_tf_dim_ordering_tf_kernels.h5"
        model_path = "./weights/mobilenet_1_0_224_tf.h5"
        #model = VGG16(weights="imagenet", include_top=True)
        #model = ResNet50(weights="imagenet", include_top=True)
        #model = ResNet50(weights=None, include_top=True)
        # keras == 2.3.1, tensorflow==1.15.0

        input_tensor = Input(shape=(224,224, 3)) # or you could put (None, None, 3) for shape
        model = MobileNet(input_tensor = input_tensor,  include_top = True, weights=None)
        model.load_weights(model_path)

        #model = load_model(model_path)

        # テスト画像の読み込みと表示
        # Jupyter Notebookのカレントディレクトリにテスト画像は置く
        image_path = "img0001.jpg"
        #image = Image.load_img(image_path, target_size=(image_rows, image_cols))


        #x = Image.img_to_array(image)
        x = cv2.resize(img, (image_cols, image_rows))
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)  # add batch size dim
        x = preprocess_input(x)

        preds = model.predict(x)
        # #print(preds) # ラベル1000個の計算結果
        results = decode_predictions(preds, top=3)[0]  # 上位3位までの結果
        # print(results)  # 上位3位までの確率とラベルを記述
        pred_name = results[0][1]
        print("この画像のクラスは", results[0][1])  # 1位の確率のラベルを記述

        return render_template('index.html', raw_img_url=raw_img_url, pred_name=pred_name)

    else:
        return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.debug = True
    app.run()
