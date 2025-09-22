import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, Callback
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from google.colab import drive

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
y = train_labels['label'].values

valtest_resnet_features = valtest_feats['resnet_feature']
valtest_vit_features = valtest_feats['vit_feature']
valtest_clip_features = valtest_feats['clip_feature']
valtest_dino_features = valtest_feats['dino_feature']
X_test = np.hstack([valtest_resnet_features, valtest_vit_features, valtest_clip_features, valtest_dino_features])

# Mixup (opsiyonel)
def mixup_data(X, y, alpha=0.2):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    lam = np.random.beta(alpha, alpha)
    X_mix = lam * X + (1 - lam) * X_shuffled
    return X_mix, y, lam, y_shuffled

# F1 Callback
class F1Callback(Callback):
    def __init__(self, X_val, y_val):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.best_f1 = 0.0

    def on_epoch_end(self, epoch, logs=None):
        y_pred_proba = self.model.predict(self.X_val)
        y_pred = np.argmax(y_pred_proba, axis=1)
        f1_val = f1_score(self.y_val, y_pred, average='macro')
        logs['val_f1'] = f1_val
        print(f"Epoch {epoch+1} => val_f1: {f1_val:.5f}")

        if f1_val > self.best_f1:
            self.best_f1 = f1_val
            print(f"Yeni en iyi F1 => {f1_val:.5f}, model kaydedildi.")
            self.model.save_weights("best_f1_model.weights.h5")

# ExponentialDecay ile LR
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=800,
    decay_rate=0.95,
    staircase=False
)

# Model Oluşturma
def create_model(input_dim):
    model = Sequential()
    model.add(Dense(1024, activation='relu', kernel_regularizer=l2(0.0001), input_shape=(input_dim,)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(512, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.0001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(10, activation='softmax'))

    # Bu satırda schedule verilince LR manuel ayarlanamaz
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Ensemble Yapısı
SEED_LIST = [42, 2023, 777]
FOLDS = 5
oof_preds = np.zeros((len(X), len(SEED_LIST)*FOLDS, 10))
test_preds = np.zeros((len(X_test), len(SEED_LIST)*FOLDS, 10))

meta_col_index = 0

for seed_val in SEED_LIST:
    print(f"\n========================== SEED = {seed_val} ==========================")
    np.random.seed(seed_val)
    tf.random.set_seed(seed_val)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=seed_val)

    for fold_index, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_index+1} --- (Seed {seed_val})")
        X_train_cv, X_val_cv = X[train_idx], X[val_idx]
        y_train_cv, y_val_cv = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_cv = scaler.fit_transform(X_train_cv)
        X_val_cv = scaler.transform(X_val_cv)

        X_train_mix, y_train_mix, _, _ = mixup_data(X_train_cv, y_train_cv, alpha=0.2)

        model = create_model(X_train_cv.shape[1])

        # Burada ReduceLROnPlateau yok
        f1_cb = F1Callback(X_val_cv, y_val_cv)
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(
            X_train_mix, y_train_mix,
            validation_data=(X_val_cv, y_val_cv),
            epochs=40,
            batch_size=128,
            callbacks=[f1_cb, early_stopping],
            verbose=1
        )

        model.load_weights("best_f1_model.weights.h5")

        # OOF
        val_proba = model.predict(X_val_cv)
        oof_preds[val_idx, meta_col_index, :] = val_proba

        # Test
        X_test_scaled = scaler.transform(X_test)
        test_proba = model.predict(X_test_scaled)
        test_preds[:, meta_col_index, :] = test_proba

        meta_col_index += 1

# Stacking (LogisticRegression)
oof_reshaped = oof_preds.reshape(len(X), -1)
test_reshaped = test_preds.reshape(len(X_test), -1)

meta_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=1000
)
meta_model.fit(oof_reshaped, y)

oof_pred_meta = meta_model.predict(oof_reshaped)
f1_ensemble = f1_score(y, oof_pred_meta, average='macro')
print("\nFinal Ensemble OOF F1 =>", f1_ensemble)

test_pred_meta = meta_model.predict(test_reshaped)
submission_df = pd.DataFrame({"ID": range(len(test_pred_meta)), "Predicted": test_pred_meta})
submission_name = f"/content/drive/My Drive/Colab_Data/submission_ensemble_stacking_F1_{f1_ensemble:.5f}.csv"
submission_df.to_csv(submission_name, index=False)
print(f"\nSubmission kaydedildi => {submission_name}")
