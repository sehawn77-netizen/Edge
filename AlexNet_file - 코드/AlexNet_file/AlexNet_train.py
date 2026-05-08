import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# GPU 메모리 설정
physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 하이퍼파라미터 설정
epochs = 10
batch_size = 64  # 배치 크기를 늘려 학습 속도 개선
learning_rate = 1e-3  # 학습률 증가로 학습 속도 조절

# AlexNet 모델 정의
def create_alexnet_model(input_shape=(32, 32, 3), num_classes=10):
    model = models.Sequential([
        # Conv Layer 1
        layers.Conv2D(96, kernel_size=3, strides=1, padding='same', activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Conv Layer 2
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Conv Layer 3, 4, 5
        layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(384, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.Conv2D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
        layers.MaxPooling2D(pool_size=2, strides=2),

        # Flatten and Fully Connected Layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),  # FC 레이어 크기 축소
        layers.Dropout(0.5),
        layers.Dense(512, activation='relu'),  # FC 레이어 크기 축소
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# CIFAR-10 데이터셋으로 AlexNet 모델 학습
def train_alexnet_on_cifar10():
    # CIFAR-10 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 데이터 전처리
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # tf.data.Dataset으로 변환 및 크기 조정
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size).map(lambda x, y: (tf.image.resize(x, (32, 32)), y)).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).map(lambda x, y: (tf.image.resize(x, (32, 32)), y)).prefetch(tf.data.AUTOTUNE)

    # AlexNet 모델 생성
    model = create_alexnet_model(
        input_shape=(32, 32, 3), 
        num_classes=10
    )

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=learning_rate), 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )

    # 콜백 설정
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # 모델 훈련
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=epochs,
        callbacks=[early_stopping]
    )

    # 모델 평가
    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}")
    
    # 모델을 저장
    model_path = os.path.join(os.getcwd(), 'alexnet_cifar10.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

# 함수 호출
if __name__ == '__main__':
    train_alexnet_on_cifar10()
