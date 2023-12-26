import cv2
import easyocr
import base64
import numpy as np
from flask import Flask, render_template, jsonify, request
from deep_translator import GoogleTranslator

app = Flask(__name__)
reader = easyocr.Reader(['id', 'en'], gpu=True)
global_image_data = None
global_text_list = None

def draw_boxes(image, result):
    for detection in result:
        bbox = detection[0]
        text = detection[1]
        cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[2][0], bbox[2][1]), (0, 255, 0), 2)
        cv2.putText(image, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

@app.route('/')
def index():
    return render_template('HOME.html')

@app.route('/image')
def image():
    return render_template('IMAGE.html')

@app.route('/text')
def text():
    return render_template('TEXT.html')

@app.route('/about')
def about():
    return render_template('ABOUT-US.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # Mendapatkan data gambar dari payload JSON
        data = request.get_json()
        image_data = data.get('image_data', '')

        # Menyimpan image_data sebagai variabel global
        global global_image_data
        global_image_data = image_data

        # Melakukan dekode base64 dan membaca gambar dengan OpenCV
        decoded_image = base64.b64decode(global_image_data.split(',')[1])
        nparr = np.frombuffer(decoded_image, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Melakukan pengolahan gambar dan mendapatkan teks
        result = reader.readtext(frame)
        text_list = [detection[1] for detection in result]
        frame = draw_boxes(frame, result)

        global global_text_list
        global_text_list = text_list

        # Memberikan respons ke klien
        response_data = {'status': 'success', 'message': 'Image processed successfully'}
        return jsonify(response_data)
    except Exception as e:
        # Menangani kesalahan jika terjadi
        print(f"Error processing image: {str(e)}")
        response_data = {'status': 'error', 'message': 'Error processing image'}
        return jsonify(response_data)

@app.route('/get_text', methods=['GET'])
def get_text():
    try:
        global global_text_list
        text_list = global_text_list

        # Memberikan respons ke klien dengan teks hasil pengolahan
        return jsonify('\n'.join(text_list))
    except Exception as e:
        # Menangani kesalahan jika terjadi
        print(f"Error processing text: {str(e)}")
        return jsonify('')

@app.route('/translate_text', methods=['POST'])
def translate_text():
    try:
        text_to_translate = request.form['text']
        source_text = request.form['source']
        target_text = request.form['target']
        translated_text = GoogleTranslator(source=source_text, target=target_text).translate(text_to_translate)
        return jsonify(translated_text)
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return jsonify('')

if __name__ == "__main__":
    app.run(debug=True)