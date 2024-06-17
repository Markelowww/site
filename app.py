from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import timm
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

# Проверка наличия GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка предварительно обученной модели Xception71
model = timm.create_model(
    'xception41.tf_in1k',
    pretrained=True,
    num_classes=7,
)

# Загрузка весов модели на выбранное устройство (CPU или GPU)
weights = torch.load('best_validation_weights.pt', map_location=device)
model.load_state_dict(weights, strict=True)
model.to(device)  # Перенос модели на устройство
model.eval()  # Перевод модели в режим оценки

# Отключение градиентов для режима оценки
torch.set_grad_enabled(False)

# Список классов заболеваний
disease_types = ['AMD', 'CSR', 'DME', 'DR', 'ERM', 'MH', 'NO']

# Преобразования для предобработки изображения
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),          # Преобразование в тензор PyTorch
    transforms.Normalize(
        mean=[0.103267140686512, 0.1032579243183136, 0.10324779152870178],
        std=[0.15830212831497192, 0.15828430652618408, 0.15826724469661713]
    )  # Нормализация изображения
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diseases')
def diseases():
    return render_template('diseases.html')

@app.route('/amd')
def amd():
    return render_template('amd.html')

@app.route('/csr')
def csr():
    return render_template('csr.html')

@app.route('/dme')
def dme():
    return render_template('dme.html')

@app.route('/dr')
def dr():
    return render_template('dr.html')

@app.route('/erm')
def erm():
    return render_template('erm.html')

@app.route('/mh')
def mh():
    return render_template('mh.html')

@app.route('/analyze_image', methods=['POST'])
def analyze_image():
    if 'photo' not in request.files:
        return redirect(request.url)
    file = request.files['photo']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        image = Image.open(filepath)
        image = image.convert('RGB')  # Убедиться, что изображение в формате RGB

        # Предобработка изображения
        img = preprocess(image)

        # Добавление размерности пакета и преобразование к типу float
        img = img.unsqueeze(0).float().to(device)  # Перенос изображения на устройство

        # Классификация с использованием модели
        predictions = model(img)

        # Обработка результатов анализа и возвращение пользователю
        predicted_class = torch.argmax(predictions, dim=1).item()
        predicted_disease = disease_types[predicted_class]
        
        image_url = url_for('static', filename='uploads/' + filename)
        return render_template('result.html', predicted_disease=predicted_disease, image_url='uploads/' + filename)
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
