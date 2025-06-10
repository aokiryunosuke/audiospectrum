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
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>音源スペクトログラム表示</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 1rem; max-width: 800px; margin: auto; }
        form { margin-bottom: 1rem; }
        img { max-width: 100%; height: auto; }
    </style>
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
    file = request.files.get('audio_file')
    if not file:
        return render_template_string(HTML_TEMPLATE, image_url=None)
    filename = file.filename
    # サニタイズ: 保存名に問題がある場合は一時名を生成
    safe_filename = os.path.basename(filename)
    filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
    file.save(filepath)

    try:
        y, sr = librosa.load(filepath, sr=None)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        plt.colorbar(format='%+2.0f dB')
        plt.title('スペクトログラム')
        plt.tight_layout()

        output_filename = f"spectrogram_{safe_filename}.png"
        output_path = os.path.join(STATIC_FOLDER, output_filename)
        plt.savefig(output_path)
        plt.close()

        image_url = '/' + output_path
    except Exception as e:
        # エラーハンドリング: 標準出力へログ
        print(f"Error processing file: {e}")
        image_url = None

    return render_template_string(HTML_TEMPLATE, image_url=image_url)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Render では 0.0.0.0 でバインドが必要
    app.run(host='0.0.0.0', port=port)
