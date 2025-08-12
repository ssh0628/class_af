import sys, os, time, shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    
from flask import Flask, Flask, render_template, request, Response, stream_with_context
from model.DBprepare import ppg_signal
from model.tf_deep import kfold, load_npz_data, group_ids

app = Flask(__name__)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.abspath(os.path.join(PROJECT_ROOT, '..'))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploaded_dataset')
SAVE_FOLDER = os.path.join(BASE_DIR, 'processed_npz')
MODEL_FOLDER = os.path.join(BASE_DIR, 'model')
PRETRAINED_MODEL_PATH = os.path.join(MODEL_FOLDER, 'deepbeat_singletask_pretrained.h5')
TF_DEEP_SCRIPT = os.path.join(MODEL_FOLDER, 'tf_deep.py')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SAVE_FOLDER'] = SAVE_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SAVE_FOLDER, exist_ok=True)

def train_model_generator():
    yield "data: Starting training...\n\n"
    time.sleep(0.01)
    
    original_stdout = sys.stdout
    
    try:
        data_dir = SAVE_FOLDER
        model_path = PRETRAINED_MODEL_PATH
        
        data, label, file_paths = load_npz_data(data_dir)
        groups = group_ids(file_paths)

        if len(data) == 0 or len(label) == 0 or len(groups) == 0:
            yield "data: [Error] One or more inputs are empty. Check your data loading and file parsing.\n\n"
        else:
            for line in kfold(data, label, groups, k=3, batch_size=64, num_classes=2, model_path=model_path):
                cleaned_line = line.strip().replace("<br>", "\n")
                yield f"data: {cleaned_line}\n\n"
                time.sleep(0.05)
                
        yield "data: Training complete.\n\n"
                
    except Exception as e:
        yield f"[ERROR] {e}\n"
    finally:
        sys.stdout = original_stdout

# HOMEPAGE
@app.route('/', methods=['GET', 'POST'])
def homepage():      
    return render_template('home.html')

# UPLOAD
@app.route('/upload', methods=['POST'])
def upload():
    uploaded_files = request.files.getlist("folder")
    
    if not uploaded_files or uploaded_files[0].filename == '':
        return "No files uploaded", 400
    
    # Clear upload folder
    if os.path.exists(UPLOAD_FOLDER):
        shutil.rmtree(UPLOAD_FOLDER)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Save files
    for file in uploaded_files:
        filename = file.filename
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        
    # Preprocessing
    try:
        ppg_signal(UPLOAD_FOLDER, SAVE_FOLDER)
        return render_template("home.html", message="Preprocessing complete")
    except Exception as e:
        return render_template("home.html", message=f"[ERROR] Processing failed: {str(e)}")
    
@app.route('/train')
def train():
    return Response(stream_with_context(train_model_generator()), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run()