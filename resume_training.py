from ultralytics import YOLO

# --- 1. Конфигурация для Возобновления Обучения ---
RESUME_MODEL_PATH = 'runs/detect/yolo11_extinguisher_FINAL_120E/weights/last.pt' 
DATA_CONFIG_PATH = 'FireExtinguisher.v2i.yolov11/data.yaml'
OUTPUT_TRAIN_PATH = 'yolo11_extinguisher_FINAL_120E' 
NEW_EPOCH_LIMIT = 100


# --- 2. Запуск Финального Обучения (Продолжение) ---
try:
    print(f"--- Загрузка модели из: {RESUME_MODEL_PATH} ---")
    model = YOLO(RESUME_MODEL_PATH)

    start_epoch = model.trainer.epochs if hasattr(model.trainer, 'epochs') else '?'
    print(f"Модель будет возобновлена с эпохи: {start_epoch + 1 if start_epoch != '?' else 'неизвестной'}")
    print(f"Финальный лимит эпох: {NEW_EPOCH_LIMIT}")

    final_results = model.train(
        data=DATA_CONFIG_PATH,
        epochs=NEW_EPOCH_LIMIT,
        name=OUTPUT_TRAIN_PATH,
        device='mps',
        patience=50
    )

    print("\n--- Финальное обучение успешно возобновлено и завершено! ---")

except FileNotFoundError:
    print(f"\n❌ Ошибка: Файл контрольной точки не найден по пути {RESUME_MODEL_PATH}.")
    print("Пожалуйста, убедитесь, что путь верен, и файл last.pt существует.")
except Exception as e:
    print(f"\n❌ Произошла ошибка при возобновлении обучения: {e}")