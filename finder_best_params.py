import os
import yaml
from ultralytics import YOLO

# --- 1. Конфигурация ---
MODEL_NAME = 'yolo11n.pt'
DATA_CONFIG_PATH = os.path.join('FireExtinguisher.v2i.yolov11', 'data.yaml')
OUTPUT_TUNE_PATH = 'yolo11_extinguisher_tune_results'

model = YOLO(MODEL_NAME)

print("--- Начинаем поиск оптимальных гиперпараметров (Тюнинг) ---")
tune_results = model.tune(
    data=DATA_CONFIG_PATH,
    epochs=10,
    imgsz=640,
    iterations=50,
    optimizer='AdamW',
    batch=32,
    device='mps',
    name=OUTPUT_TUNE_PATH
)

print("\n--- Тюнинг завершен! ---")

# --- 3. Сохранение лучших параметров в файл ---
best_hyp = tune_results.best_hyp
hyp_file_path = 'best_hyp_extinguisher.yaml'

with open(hyp_file_path, 'w') as f:
    yaml.dump(best_hyp, f, sort_keys=False)

print(f"\n✅ Найденные лучшие параметры сохранены в файле: {hyp_file_path}")
