!pip install lightgbm

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from google.colab import drive

# LightGBM Meta-Learner
from lightgbm import LGBMClassifier

# 1) Google Drive
drive.mount('/content/drive')

# 2) Dosya Yolları
train_feats_path = '/content/drive/My Drive/Colab_Data/train_feats.npy'
train_labels_path = '/content/drive/My Drive/Colab_Data/train_labels.csv'
valtest_feats_path = '/content/drive/My Drive/Colab_Data/valtest_feats.npy'

# 3) Verileri Yükle
train_feats = np.load(train_feats_path, allow_pickle=True).item()
train_labels = pd.read_csv(train_labels_path)
valtest_feats = np.load(valtest_feats_path, allow_pickle=True).item()

resnet_features = train_feats['resnet_feature']
vit_features = train_feats['vit_feature']
clip_features = train_feats['clip_feature']
dino_features = train_feats['dino_feature']

X = np.hstack([resnet_features, vit_features, clip_features, dino_features])
y_raw = train_labels['label'].values  # Ham (integer) etiketler

valtest_resnet_features = valtest_feats['resnet_feature']
valtest_vit_features = valtest_feats['vit_feature']
valtest_clip_features = valtest_feats['clip_feature']
valtest_dino_features = valtest_feats['dino_feature']
X_test = np.hstack([valtest_resnet_features, valtest_vit_features, valtest_clip_features, valtest_dino_features])

num_classes = 10  # 10 sınıf olduğunu varsaydık

# 4) Soft Label Mixup Fonksiyonu
def mixup_data(X_in, y_in_onehot, alpha=0.2):
    indices = np.random.permutation(len(X_in))
    X_shuffled = X_in[indices]
    y_shuffled_onehot = y_in_onehot[indices]
    lam = np.random.beta(alpha, alpha)
    X_mix = lam * X_in + (1 - lam) * X_shuffled
    y_mix = lam * y_in_onehot + (1 - lam) * y_shuffled_onehot  # soft label
    return X_mix, y_mix

# 5) F1 Callback - Soft Label Versiyon
class F1Callback(Callback):
    def __init__(self, X_val, y_val_onehot):
        super().__init__()
        self.X_val = X_val
        self.y_val_onehot = y_val_onehot  # One-hot validation etiketleri
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        # Tahmin (softmax)
        y_pred_proba = self.model.predict(self.X_val)
        # y_val_onehot => shape [val_size, num_classes]
        # Convert them to integer classes
        true_classes = np.argmax(self.y_val_onehot, axis=1)
        pred_classes = np.argmax(y_pred_proba, axis=1)

        f1_val = f1_score(true_classes, pred_classes, average='macro')
        logs['val_f1'] = f1_val
        print(f"Epoch {epoch+1} => val_f1: {f1_val:.5f}")

        if f1_val > self.best_f1:
            self.best_f1 = f1_val
            print(f"Yeni en iyi F1 => {f1_val:.5f}, model kaydedildi.")
            self.model.save_weights("best_f1_model.weights.h5")

# 6) Exponential Decay - LR
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=5e-4,  # Daha yavaş başlangıç LR
    decay_steps=1000,
    decay_rate=0.95,
    staircase=False
)

# 7) Model Oluşturma (düşük dropout, ufak L2)
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.00005), input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.00005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.00005)))
    model.add(BatchNormalization())
    model.add(Dropout(0.15))

    model.add(Dense(num_classes, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    # Soft label => 'categorical_crossentropy'
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 8) Büyük Ensemble (5 Seeds x 5 Folds = 25 Model)
SEED_LIST = [42, 2023, 777, 999, 314159]
FOLDS = 5

# OOF ve Test tahminlerini saklayacak
oof_preds = np.zeros((len(X), len(SEED_LIST)*FOLDS, num_classes))
test_preds = np.zeros((len(X_test), len(SEED_LIST)*FOLDS, num_classes))

meta_col_index = 0

for seed_val in SEED_LIST:
    print(f"\n=================== SEED = {seed_val} ===================")
    np.random.seed(seed_val)
    tf.random.set_seed(seed_val)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed_val)

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(X, y_raw)):
        print(f"\n--- Fold {fold_index+1} / Seed {seed_val} ---")

        # Ayrıştırma
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y_raw[train_idx], y_raw[val_idx]

        # One-hot
        y_train_cv_onehot = tf.keras.utils.to_categorical(y_train_cv, num_classes=num_classes)
        y_val_cv_onehot = tf.keras.utils.to_categorical(y_val_cv, num_classes=num_classes)

        # Ölçekleme
        scaler = StandardScaler()
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_val_cv = scaler.transform(X_val_cv)

        # Mixup (soft labels)
        X_train_mix, y_train_mix = mixup_data(X_train_cv, y_train_cv_onehot, alpha=0.2)

        # Model
        model = create_model(X_train_cv.shape[1])

        # Callback
        f1_cb = F1Callback(X_val_cv, y_val_cv_onehot)
        early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

        # Eğit
        model.fit(
            X_train_mix, y_train_mix,
            validation_data=(X_val_cv, y_val_cv_onehot),
            epochs=50,
            batch_size=64,
            callbacks=[f1_cb, early_stopping],
            verbose=1
        )

        # En iyi modelin weights'ini yükle
        model.load_weights("best_f1_model.weights.h5")

        # OOF
        val_proba = model.predict(X_val_cv)
        oof_preds[val_idx, meta_col_index, :] = val_proba

        # Test
        X_test_scaled = scaler.transform(X_test)
        test_proba = model.predict(X_test_scaled)
        test_preds[:, meta_col_index, :] = test_proba

        meta_col_index += 1

# 9) Meta-Özellikleri Hazırla
oof_reshaped = oof_preds.reshape(len(X), -1)
test_reshaped = test_preds.reshape(len(X_test), -1)

# 10) LightGBM Meta-Learner
meta_model = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=-1,
    random_state=42
)

# OOF verisi -> meta model
meta_model.fit(oof_reshaped, y_raw)

oof_pred_meta = meta_model.predict(oof_reshaped)
f1_ensemble = f1_score(y_raw, oof_pred_meta, average='macro')
print(f"\nFinal Ensemble OOF F1 => {f1_ensemble:.5f}")

# Test son tahmin
test_pred_meta = meta_model.predict(test_reshaped)
submission_df = pd.DataFrame({"ID": range(len(test_pred_meta)), "Predicted": test_pred_meta})
submission_name = f"/content/drive/My Drive/Colab_Data/submission_big_ensemble_lgbm_F1_{f1_ensemble:.5f}.csv"
submission_df.to_csv(submission_name, index=False)
print(f"\nSubmission kaydedildi => {submission_name}")