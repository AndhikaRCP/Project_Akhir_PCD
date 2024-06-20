from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
import os

app = Flask(__name__ ,template_folder="pages")
app.config['UPLOAD_FOLDER'] = 'uploads'

uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

def save_image(image):
    filename = secure_filename(image.filename)
    image.save(os.path.join(uploads_dir, filename))
    return filename 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # handle POST request
        file = request.files['image']
        print(file)
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return render_template('result.html', filename=filename)
    else:
        # handle GET request
        return render_template('upload.html')  # or any other template

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)