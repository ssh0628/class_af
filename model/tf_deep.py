import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, sys, io, re
from keras.models import load_model, Model
from keras.layers import Dense
from keras.losses import Loss
# from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 현재 파일(__file__)이 있는 폴더 경로
current_dir = os.path.dirname(os.path.abspath(__file__))

# 그 상위 폴더 경로
parent_dir = os.path.dirname(current_dir)

""" import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--model_path', type=str, required=True)
args = parser.parse_args() """

""" data_dir = args.data_dir
model_path = args.model_path """

""" print(f"Using data_dir: {data_dir}")
print(f"Using model_path: {model_path}") """

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

#signal, label

class FocalLoss(Loss):
    def __init__(self, gamma=2.0, alpha=0.85, from_logits=False, reduction='sum_over_batch_size', name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits
        
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
            
        eps = 1e-4
        y_pred = tf.clip_by_value(y_pred, eps, 1 - eps)

        # tf.print("y_true:", y_true)
        # tf.print("y_pred:", y_pred)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1-y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1-self.alpha)
        
        focal_weight = alpha_factor * tf.pow(1 - pt, self.gamma)
        
        loss = -focal_weight * tf.math.log(pt)
        
        return tf.reduce_mean(loss)
        
def load_npz_data(dir_path):
    data = []
    labels = []
    file_paths = []
    dropped_count = 0
    
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            path = os.path.join(root, filename)
            try:
                with np.load(path) as npz:
                    ppg = npz['ppg']
                    label = npz['label']
                    
                    if np.isnan(ppg).any():
                        print(f"[Warning] NaN found in {filename}, dropping this sample.", flush=True)
                        dropped_count += 1
                        continue
                    
                    data.append(ppg)
                    labels.append(label)
                    file_paths.append(path)
                    
            except Exception as e:
                print(f"[Error] Failed to load {filename}: {e}", flush=True)
                
    print(f"Dropped samples due to NaN: {dropped_count}", flush=True)        
    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32) 
    # labels = to_categorical(labels, num_classes=2)
    data = data[..., np.newaxis]
    
    # print(f"Data shape: {data.shape}, label shape: {labels.shape}")
    # print(f"Data range: min={np.min(data)}, max={np.max(data)}")
    # print(f"Labels unique: {np.unique(labels)}")
    # print(f"NaN count in data: {np.isnan(data).sum()}")
    # print(f"NaN count in labels: {np.isnan(labels).sum()}")

    
    return data, labels, file_paths

""" def create_datasets(data, batch_size=64, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices(data)
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(data))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset """

def set_model(model_path, num_classes, fold_num=None):
    base_model = load_model(model_path, compile=False)
    
    for layer in base_model.layers:
        layer.trainable = True
    
    x = base_model.get_layer(index=-2).output  
    dense_name = f'new_dense_fold{fold_num}' if fold_num is not None else 'new_dense'

    new_output = Dense(1, activation=None, name=dense_name)(x)
    model = Model(inputs=base_model.input, outputs = new_output)
    
    return model

def get_id(filename):
    match = re.search(r'(af|non_af)_\d{3}', filename.lower())
    
    if match:
        return match.group()
    else:
        raise ValueError(f"Invalid filename format: {filename}")

def group_ids(file_paths):
    groups = []
    
    for path in file_paths:
        parent_folder = os.path.basename(os.path.dirname(path))  # af_001
        groups.append(parent_folder)
        
    """ for filename in os.listdir(dir_path):
        if filename.endswith('.npz'):
            groups.append(get_id(filename))"""
            
    return np.array(groups)
    
def kfold(data, labels, groups, k=3, batch_size=64, num_classes=2, model_path=None):
    # kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 1
    all_scores = []
    fold_logs = []
    all_fold_histories = []
    
    best_model = None
    best_fold = -1
    best_f1 = -1
    
    for train_index, valid_index in sgkf.split(data, labels, groups):
        # print(f"[Log]Fold : {fold}", flush=True)
        yield f"[Log] Fold {fold} started\n"
        
        x_train, x_valid = data[train_index], data[valid_index]
        y_train, y_valid = labels[train_index], labels[valid_index]
        y_train = y_train.astype(np.float32).reshape(-1, 1)
        y_valid = y_valid.astype(np.float32).reshape(-1, 1)

        # print(f"Train label distribution: {np.unique(y_train, return_counts=True)}")
        # print(f"Valid label distribution: {np.unique(y_valid, return_counts=True)}")
        # print(np.unique(y_train))  # [0. 1.] 이렇게만 나와야 함
        # print(np.isnan(y_train).sum())  # 0이어야 정상
        # print(y_train.dtype, y_train.shape)
        # print(np.min(y_train), np.max(y_train))  # 0~1 사이인지
        # print(np.isnan(x_train).sum(), np.isnan(x_valid).sum())
        # print(np.isnan(data).sum(), np.isinf(data).sum())
        # print(np.isnan(label).sum(), np.isinf(label).sum())
        # print(f"Data range: min={data.min()}, max={data.max()}")
        
        model = set_model(model_path, num_classes, fold_num=fold)
        
        model.compile(
            optimizer = Adam(learning_rate=1e-4), 
            # loss = tf.keras.losses.BinaryCrossentropy(), 
            loss = FocalLoss(gamma=2.0, alpha=0.85, from_logits=True), 
            metrics=['accuracy']
        )
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ModelCheckpoint(f"best_fold{fold}.keras", save_best_only=True, monitor='val_loss', mode='min')
        ]

        history = model.fit(
            x_train, y_train, 
            validation_data = (x_valid, y_valid), 
            epochs = 100, 
            batch_size=batch_size, 
            callbacks=callbacks
        )
        
        all_fold_histories.append({
        'fold': fold,
        'train_loss': history.history['loss'],
        'train_acc': history.history['accuracy'],
        'val_loss': history.history['val_loss'],
        'val_acc': history.history['val_accuracy']
        })

        val_acc = history.history.get('val_accuracy', [])[-1]
        val_loss = history.history.get('val_loss', [])[-1]
        train_acc = history.history.get('accuracy', [])[-1]
        train_loss = history.history.get('loss', [])[-1]
        
        
        y_pred = model.predict(x_valid)
        y_pred_binary = (y_pred > 0.5).astype(int) # y_pred_binary = np.round(y_pred).astype(int)
        
        f1 = f1_score(y_valid.reshape(-1), y_pred_binary.reshape(-1), average='binary') # 클래스 불균형이 심할땐 average='macro'
        
        fold_logs.append({
            'fold': fold,
            'train_acc': train_acc,
            'train_loss': train_loss,
            'val_acc': val_acc,
            'val_loss': val_loss,
            'f1': f1
        })
        
        #print(f"[Result] Fold {fold} - F1-score: {f1:.4f}")
        yield f"[Result] Fold {fold} F1-score: {f1:.4f}\n"
        all_scores.append(f1)
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_fold = fold
            # print(f"[Info] New best model found at Fold {fold} with F1: {f1:.4f}")
            yield f"[Info] New best model found at Fold {fold} with F1: {f1:.4f}\n"
        fold += 1
        
    #print("\n[Training Summary per Fold]")
    yield "\n[Training Summary per Fold]\n"
    for log in fold_logs:
        # print(f"Fold {log['fold']}: Train Acc={log['train_acc']:.4f}, Train Loss={log['train_loss']:.4f}, Val Acc={log['val_acc']:.4f}, Val Loss={log['val_loss']:.4f}, F1={log['f1']:.4f}", flush=True)
        yield f"Fold {log['fold']}: Train Acc={log['train_acc']:.4f}, Train Loss={log['train_loss']:.4f}, Val Acc={log['val_acc']:.4f}, Val Loss={log['val_loss']:.4f}, F1={log['f1']:.4f}\n"
    if best_model:
        # save_path = rf"C:\Users\cream\OneDrive\Desktop\neuro\model\best_model_f1_fold{best_fold}.keras"
        save_path = model_path
        save_path = model_path.replace('.h5', '.keras')
        # best_model.save(save_path)
        # print(f"\n[Save] Best model saved as: {save_path}")
        yield f"\n[Save] Best model saved as: {save_path}\n"

    avg_f1 = np.mean(all_scores)
    # print(f"\n[Summary] Average F1-score: {avg_f1:.4f}")
    yield f"\n[Summary] Average F1-score: {avg_f1:.4f}\n"
    
    plot_dir = os.path.join(parent_dir, 'plots')
    # 폴더 없으면 만들기
    os.makedirs(plot_dir, exist_ok=True)
    
    for hist in all_fold_histories:
        fold = hist['fold']
        epochs = range(1, len(hist['train_loss']) + 1)

        plt.figure(figsize=(12, 5))

        # Loss plot
        plt.subplot(1, 2, 1)
        plt.plot(epochs, hist['train_loss'], label='Train Loss')
        plt.plot(epochs, hist['val_loss'], label='Val Loss')
        plt.title(f'Fold {fold} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Accuracy plot
        plt.subplot(1, 2, 2)
        plt.plot(epochs, hist['train_acc'], label='Train Acc')
        plt.plot(epochs, hist['val_acc'], label='Val Acc')
        plt.title(f'Fold {fold} Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        save_path = os.path.join(plot_dir, f'fold_{fold}_training_plot.png')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
       
    """ print("Average: ")
    avg_score = np.mean(all_scores, axis=0)
    print(f"AverageLoss : {avg_score[0]:.4f}, Average Accuracy : {avg_score[1]:.4f}") """
    
if __name__ == "__main__":
    
    # data_dir = r"C:\Users\cream\OneDrive\Desktop\neuro\DB3"
    # model_path = r"C:\Users\cream\OneDrive\Desktop\neuro\model\deepbeat_singletask_pretrained.h5"

    """ data, label, file_paths = load_npz_data(data_dir)
    groups = group_ids(file_paths)
    # config
    batch_size = 64
    CLASS = ['AF', 'NON_AF']
    num_classes = len(CLASS)
    
    print(f"Data shape: {data.shape}")
    print(f"Label shape: {label.shape}")
    print(f"Groups shape: {groups.shape}")
    print(f"Unique labels: {np.unique(label)}")
    print(f"Unique groups: {np.unique(groups)}")

    if len(data) == 0 or len(label) == 0 or len(groups) == 0:
        print("[Error] One or more inputs are empty. Check your data loading and file parsing.")
    else:
        batch_size = 64
        CLASS = ['AF', 'NON_AF']
        num_classes = len(CLASS)
        
         # training
        kfold(data, label, groups, k=3, batch_size=batch_size, num_classes=num_classes, model_path=model_path) """