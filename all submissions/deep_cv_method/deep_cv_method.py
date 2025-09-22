# --- 1. Gerekli kütüphaneleri yükle ---
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# Google Drive bağlantısı
from google.colab import drive
drive.mount('/content/drive')

# Dosya yolları
train_feats_path = '/content/drive/My Drive/Colab_Data/train_feats.npy'
train_labels_path = '/content/drive/My Drive/Colab_Data/train_labels.csv'
valtest_feats_path = '/content/drive/My Drive/Colab_Data/valtest_feats.npy'

# Veriyi yükle
train_feats = np.load(train_feats_path, allow_pickle=True).item()
train_labels = pd.read_csv(train_labels_path)
valtest_feats = np.load(valtest_feats_path, allow_pickle=True).item()

# Özellikleri birleştir
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

# Model oluşturma fonksiyonu (fonksiyonel hale getirdim, böylece cross-validation'da tekrar kullanabiliriz)
def create_model(input_dim):
    model = Sequential([
        Dense(1024, activation='relu', input_dim=input_dim, kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),  # Daha düşük dropout ile başla
        Dense(512, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Biraz daha küçük LR
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
best_models = []
for train_index, val_index in skf.split(X, y):
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Scale işlemi
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    model = create_model(X_train.shape[1])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=128,
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    y_val_pred = np.argmax(model.predict(X_val), axis=1)
    fold_f1 = f1_score(y_val, y_val_pred, average='macro')
    cv_scores.append(fold_f1)
    best_models.append(model)

print("CV F1 Skorları:", cv_scores)
print("Ortalama CV F1:", np.mean(cv_scores))

# Tüm veri üzerinde son modeli eğitme (opsiyonel)
# Burada en iyi modeli seçmek istersen ek işlem gerekebilir
final_scaler = StandardScaler()
X_scaled = final_scaler.fit_transform(X)
final_model = create_model(X_scaled.shape[1])
final_model.fit(
    X_scaled, y,
    epochs=50,
    batch_size=128,
    verbose=1
)

# Test verisini tahminle
X_test_scaled = final_scaler.transform(X_test)
test_preds = np.argmax(final_model.predict(X_test_scaled), axis=1)

submission_name = f'submission_DeepCV.csv'
submission = pd.DataFrame({'ID': range(len(test_preds)), 'Predicted': test_preds})
submission_path = f'/content/drive/My Drive/Colab_Data/{submission_name}'
submission.to_csv(submission_path, index=False)
print(f"Submission dosyası {submission_path} olarak kaydedildi. 🤖")
