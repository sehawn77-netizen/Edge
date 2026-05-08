import os
import tensorflow as tf
import argparse
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

# 학습된 모델 파일 경로
TRAINED_MODEL = "/root/mobilenetv2_cifar10.h5"

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="MobileNet inference with GPU memory limitation and CPU core limitation."
    )
    parser.add_argument(
        "--gpu", type=bool, default=False, help="Use GPU for inference. (Default = True)"
    )
    parser.add_argument(
        "--gpu_mem_limit",
        type=int,
        default=512,
        help="Limit GPU memory usage in MB. Set to 0 for no limit. (Default = 512)",
    )
    parser.add_argument(
        "--memory_growth",
        type=bool,
        default=False,
        help="Enable GPU memory growth. (Default = False)",
    )
    parser.add_argument(
        "--cpu_cores",
        type=int,
        default=1,
        help="Limit the number of CPU cores for TensorFlow. Set to None for no limit.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference. (Default = 32)",
    )
    return parser.parse_args()

args = parse_args()

# Configurations
USE_GPU = args.gpu
GPU_MEM_LIMIT = args.gpu_mem_limit
MEMORY_GROWTH = args.memory_growth
CPU_CORES = args.cpu_cores
BATCH_SIZE = args.batch_size

# GPU Setup
def setup_gpu(use_gpu=False, memory_growth=False, gpu_mem_limit=512):
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU, CPU only
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print("Number of GPUs Available: {}".format(len(gpus)))
            for gpu in gpus:
                if memory_growth:
                    tf.config.experimental.set_memory_growth(gpu, True)
                if gpu_mem_limit > 0:
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=gpu_mem_limit)],
                    )

# CPU Setup (Limit CPU cores)
def setup_cpu(cpu_cores=1):
    if cpu_cores:
        tf.config.threading.set_intra_op_parallelism_threads(cpu_cores)
        tf.config.threading.set_inter_op_parallelism_threads(cpu_cores)
        print("Limiting TensorFlow to {} CPU cores.".format(cpu_cores))
    else:
        print("Using all available CPU cores.")

# GPU와 CPU 설정 실행
setup_gpu(USE_GPU, MEMORY_GROWTH, GPU_MEM_LIMIT)
setup_cpu(CPU_CORES)

# Load and Prepare Data (Batch Processing)
def load_and_prepare_data(batch_size=32):
    from tensorflow.keras.datasets import cifar10

    _, (x_test, y_test) = cifar10.load_data()
    x_test = x_test / 255.0  # 정규화

    def data_generator(x, y, batch_size):
        for i in range(0, len(x), batch_size):
            with tf.device('/CPU:0'):  # CPU에서 리사이즈 실행
                batch_x = tf.image.resize(x[i:i + batch_size], (96, 96))
            yield batch_x, y[i:i + batch_size]

    return data_generator(x_test, y_test, batch_size)

# 데이터 생성기 생성
test_data_gen = load_and_prepare_data(BATCH_SIZE)

# Load Model
try:
    loaded_model = load_model(TRAINED_MODEL, compile=False)  # Compile 설정 끔
    print("Model loaded successfully without optimizer.")
    
    # Recompile the model
    loaded_model.compile(optimizer=Adam(learning_rate=0.001),
                         loss="sparse_categorical_crossentropy",
                         metrics=["accuracy"])
    print("Model recompiled successfully.")
except Exception as e:
    print(f"Error loading or recompiling model: {e}")
    exit(1)

# 추론 시간 측정
start_time = time.time()
results = []

for batch_x, batch_y in test_data_gen:
    results.append(loaded_model.evaluate(batch_x, batch_y, verbose=0))

end_time = time.time()

# 배치별로 얻은 평가 지표 평균 계산
test_loss = sum([res[0] for res in results]) / len(results)
test_acc = sum([res[1] for res in results]) / len(results)

print("Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}".format(test_loss, test_acc))
print("Time taken for inference: {:.2f} seconds".format(end_time - start_time))
