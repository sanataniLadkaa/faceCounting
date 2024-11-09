import os
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image, ImageDraw
from retinaface import RetinaFace

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
PROCESSED_FOLDER = 'static/processed/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = PROCESSED_FOLDER

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Process the image with RetinaFace
        orgimg = Image.open(file_path).convert("RGB")
        resp = RetinaFace.detect_faces(file_path)
        draw = ImageDraw.Draw(orgimg)
        
        # Count the number of detected faces
        num_faces = 0
        
        # Loop through detected faces and draw boxes and landmarks
        for face_key, face_data in resp.items():
            facial_area = face_data['facial_area']
            draw.rectangle(facial_area, outline="red", width=2)
            landmarks = face_data['landmarks']
            num_faces += 1
            for landmark in landmarks.values():
                draw.ellipse((landmark[0] - 5, landmark[1] - 5, landmark[0] + 5, landmark[1] + 5), fill="blue")
        
        # Save the processed image
        processed_path = os.path.join(app.config['PROCESSED_FOLDER'], 'processed_' + file.filename)
        orgimg.save(processed_path)
        
        # Render the result.html template with filename and number of faces
        return render_template('result.html', filename='processed_' + file.filename, num_faces=num_faces)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return render_template('result.html', filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
