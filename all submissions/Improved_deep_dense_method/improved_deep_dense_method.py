# Gerekli kütüphaneleri yükle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score

# --- 1. Google Drive Bağlantısı ---
from google.colab import drive
drive.mount('/content/drive')

# --- 2. Dosya Yolları ---
train_feats_path = '/content/drive/My Drive/Colab_Data/train_feats.npy'
train_labels_path = '/content/drive/My Drive/Colab_Data/train_labels.csv'
valtest_feats_path = '/content/drive/My Drive/Colab_Data/valtest_feats.npy'

# --- 3. Veriyi Yükle ---
train_feats = np.load(train_feats_path, allow_pickle=True).item()
train_labels = pd.read_csv(train_labels_path)
valtest_feats = np.load(valtest_feats_path, allow_pickle=True).item()

# Eğitim verisinden özellik vektörlerini ayır
resnet_features = train_feats['resnet_feature']
vit_features = train_feats['vit_feature']
clip_features = train_feats['clip_feature']
dino_features = train_feats['dino_feature']

X = np.hstack([resnet_features, vit_features, clip_features, dino_features])  # Özellik vektörlerini birleştir
y = train_labels['label'].values  # Etiketler

# Test verisinden özellik vektörlerini ayır
valtest_resnet_features = valtest_feats['resnet_feature']
valtest_vit_features = valtest_feats['vit_feature']
valtest_clip_features = valtest_feats['clip_feature']
valtest_dino_features = valtest_feats['dino_feature']

X_test = np.hstack([valtest_resnet_features, valtest_vit_features, valtest_clip_features, valtest_dino_features])

# --- 4. Veriyi Hazırlama ---
# Veriyi eğitim ve doğrulama olarak ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Veriyi ölçeklendir
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# --- 5. Modeli Tanımlama ---
# Daha karmaşık model
model_name = "Improved_Deep_Dense_L2_BN_Ensemble"
model = Sequential([
    Dense(2048, activation='relu', input_dim=X_train.shape[1], kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1024, activation='relu', kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(512, activation='relu', kernel_regularizer=l2(0.002)),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')  # 10 sınıf olduğu için çıkış katmanında 10 nöron
])

# Modeli derleme
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- 6. Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)

# --- 7. Modeli Eğitme ---
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr]
)

# --- 8. Doğrulama Performansı ---
# Doğrulama verisinde tahmin yap
y_val_pred = np.argmax(model.predict(X_val), axis=1)
macro_f1 = f1_score(y_val, y_val_pred, average='macro')
print(f"Validation Macro F1 Score: {macro_f1}")

# --- 9. Test Verisi Tahmini ve Kaggle Formatı ---
test_predictions = np.argmax(model.predict(X_test), axis=1)

# Kaggle formatında submission dosyası hazırlama
submission_name = f'submission_{model_name}_f1_{macro_f1:.5f}.csv'
submission = pd.DataFrame({
    'ID': range(len(test_predictions)),
    'Predicted': test_predictions
})

submission_path = f'/content/drive/My Drive/Colab_Data/{submission_name}'
submission.to_csv(submission_path, index=False)
print(f"Submission dosyası {submission_path} olarak kaydedildi.")
