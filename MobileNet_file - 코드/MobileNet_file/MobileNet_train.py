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

epochs = 20
batch_size = 32  # 배치 크기를 32로 설정하여 메모리 최적화

# MobileNet-V2의 Inverted Residual Block
def inverted_residual_block(x, filters, stride, expansion=6):
    input_channels = x.shape[-1]

    # 1x1 convolution for expansion
    x = layers.Conv2D(filters=filters * expansion, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)  # ReLU6 activation

    # Depthwise separable convolution
    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)  # ReLU6 activation

    # 1x1 convolution for projection (reducing the dimensionality)
    x = layers.Conv2D(filters=filters, kernel_size=1, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)

    # Add residual connection if input and output channels are the same
    if input_channels == filters and stride == 1:
        x = layers.Add()([x, x])  # Skip connection only for matching channels and stride
    
    return x
    
# MobileNet-V2 모델 생성
def create_mobilenetv2_model(input_shape=(96, 96, 3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution layer
    x = layers.Conv2D(32, kernel_size=3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(max_value=6)(x)

    # Add several inverted residual blocks
    x = inverted_residual_block(x, filters=16, stride=1)
    x = inverted_residual_block(x, filters=24, stride=2)
    x = inverted_residual_block(x, filters=32, stride=2)

    # Global Average Pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Fully connected layer
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create and compile the model
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return model
    
# CIFAR-10 데이터셋으로 MobileNetV2 모델 학습
def train_mobilenet_on_cifar10():
    # CIFAR-10 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 데이터 전처리
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # tf.data.Dataset으로 변환 및 크기 조정
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(10000).batch(batch_size).map(lambda x, y: (tf.image.resize(x, (96, 96)), y)).prefetch(tf.data.AUTOTUNE)

    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_ds = test_ds.batch(batch_size).map(lambda x, y: (tf.image.resize(x, (96, 96)), y)).prefetch(tf.data.AUTOTUNE)

    # MobileNetV2 모델 생성
    model = create_mobilenetv2_model(
        input_shape=(96, 96, 3), 
        num_classes=10
    )

    # 모델 레이어 중 일부만 훈련 가능하도록 설정 (미세 조정)
    for layer in model.layers[-5:]:  # 마지막 5개 레이어를 학습 가능하도록 설정
        layer.trainable = True

    # 모델 컴파일
    model.compile(
        optimizer=Adam(learning_rate=1e-5), 
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
    print(f"Test Loss: {test_loss}, Test accuracy: {test_acc}")
    
    # 모델을 저장
    model_path = os.path.join(os.getcwd(), 'mobilenetv2_cifar10.h5')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Model saved to {model_path}")

# 함수 호출
if __name__ == '__main__':
    train_mobilenet_on_cifar10()
