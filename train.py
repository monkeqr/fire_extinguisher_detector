from ultralytics import YOLO

# --- 1. Конфигурация ---
MODEL_NAME = 'yolo11n.pt'
DATA_CONFIG_PATH = 'FireExtinguisher.v2i.yolov11/data.yaml'
OUTPUT_TRAIN_PATH = 'yolo11_extinguisher_FINAL_120E'

model = YOLO(MODEL_NAME)

# Мы вставляем их вручную для чистоты, хотя можно загрузить их из файла yaml.
BEST_HYPS = {
    'lr0': 0.00678,
    'lrf': 0.00544,
    'momentum': 0.937,
    'weight_decay': 0.00055,   # Регуляризация
    'warmup_epochs': 4,
    'warmup_momentum': 0.88984,
    'box': 7.8399,             # Box Loss
    'cls': 0.41339,            # Class Loss
    'dfl': 1.5,
    'hsv_h': 0.015,            # Hue
    'hsv_s': 0.62147,          # Saturation
    'hsv_v': 0.42707,          # Value
    'translate': 0.11035,      # Translate
    'scale': 0.6127,           # Scale
    'close_mosaic': 8,       # Эпохи, когда отключается аугментация Mosaic
    
    # Не найденные в тюнинге, но важные параметры:
    'degrees': 0.0,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,
    'mosaic': 1.0,
    'mixup': 0.0,
    'cutmix': 0.0,
    'copy_paste': 0.0,
}

final_results = model.train(
    data=DATA_CONFIG_PATH,
    epochs=120,
    imgsz=640,
    batch=32,
    device='mps',
    name=OUTPUT_TRAIN_PATH,
    patience=50,
    **BEST_HYPS
)
