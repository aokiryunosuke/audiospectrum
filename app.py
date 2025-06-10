from flask import Flask, request, render_template_string
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <title>音源スペクトログラム表示</title>
</head>
<body>
    <h1>音源をアップロードしてスペクトログラムを表示</h1>
    <form method="post" enctype="multipart/form-data" action="/upload">
        <input type="file" name="audio_file" accept="audio/*" required>
        <button type="submit">アップロードして解析</button>
    </form>
    {% if image_url %}
    <h2>スペクトログラム:</h2>
    <img src="{{ image_url }}" alt="Spectrogram">
    {% endif %}
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, image_url=None)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['audio_file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    y, sr = librosa.load(filepath)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('スペクトログラム')
    plt.tight_layout()

    output_path = os.path.join(STATIC_FOLDER, 'spectrogram.png')
    plt.savefig(output_path)
    plt.close()

    return render_template_string(HTML_TEMPLATE, image_url='/' + output_path)

if __name__ == '__main__':
    app.run(debug=True)
