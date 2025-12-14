from ultralytics import YOLO

# --- Конфигурация ---
# Путь к лучшим весам, которые были получены после 120 эпох
MODEL_PATH = 'runs/detect/yolo11_extinguisher_FINAL_120E2/weights/best.pt' 

# Пути к изображениям для тестирования.
IMAGE_WITH_OBJECT = 'fire_destinguisher.png'
IMAGE_WITHOUT_OBJECT = 'image.png'


# 1. Загрузка обученной модели
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Модель успешно загружена из: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл модели не найден по пути {MODEL_PATH}. Убедитесь, что обучение завершено и путь верен.")
    exit()


# 2. Тестирование на изображении с объектом
print("\n--- Тестирование на изображении с огнетушителем ---")
results_with = model.predict(
    source=IMAGE_WITH_OBJECT, 
    conf=0.25,     # Минимальная уверенность (Confidence) для детекции
    iou=0.7,       # Порог IoU для Non-Maximum Suppression
    device='mps',  # Ускорение Apple Silicon
    save=True      # Сохранить результат с боксами
)
# Результаты будут сохранены в runs/detect/predict/
print(f"Результат сохранен в папку: {results_with[0].save_dir}")


# 3. Тестирование на изображении без объекта (для демонстрации "отсутствия ложных срабатываний")
print("\n--- Тестирование на изображении БЕЗ огнетушителя ---")
results_without = model.predict(
    source=IMAGE_WITHOUT_OBJECT, 
    conf=0.25,
    iou=0.7,
    device='mps',
    save=True
)
# Результаты будут сохранены в runs/detect/predict2/ (или следующая по счету папка)
print(f"Результат сохранен в папку: {results_without[0].save_dir}")

print("\n--- Проверка работы модели на изображениях завершена ---")