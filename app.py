# app.py
import cv2
import easyocr
from flask import Flask, render_template, Response, jsonify, request
from deep_translator import GoogleTranslator
import atexit

app = Flask(__name__)
reader = easyocr.Reader(['id', 'en'], gpu=True)
latest_frame = None  # Global variable to store the latest frame
cap = cv2.VideoCapture(0)  # Initialize video capture outside the function

# Ensure that the video capture is released when the application exits
atexit.register(lambda: cap.release())

def draw_boxes(image, result):
    for detection in result:
        bbox = detection[0]
        text = detection[1]
        cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[2][0], bbox[2][1]), (0, 255, 0), 2)
        cv2.putText(image, text, (bbox[0][0], bbox[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image

def gen_frames():
    global latest_frame  # Use the global variable
    while True:
        success, frame = cap.read()
        if not success:
            break

        try:
            result = reader.readtext(frame)
            text = [detection[1] for detection in result]
            frame = draw_boxes(frame, result)
            latest_frame = frame  # Update the global variable with the latest frame
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_text')
def get_text():
    global latest_frame  # Use the global variable
    while True:
        success, frame = cap.read()
        if not success:
            break

        try:
            result = reader.readtext(frame)
            text_list = [detection[1] for detection in result]
            frame = draw_boxes(frame, result)
            latest_frame = frame  # Update the global variable with the latest frame
            return jsonify('\n'.join(text_list))
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return jsonify('')

@app.route('/translate_text', methods=['POST'])
def translate_text():
    try:
        text_to_translate = request.form['text']
        translated_text = GoogleTranslator(source='id', target='en').translate(text_to_translate)
        return jsonify(translated_text)
    except Exception as e:
        print(f"Error translating text: {str(e)}")
        return jsonify('')

if __name__ == "__main__":
    app.run(debug=True)