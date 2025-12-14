from ultralytics import YOLO
import time
import os
import cv2

# --- Конфигурация проекта и путей ---
MODEL_PATH = 'runs/detect/yolo11_extinguisher_FINAL_120E2/weights/best.pt'
SOURCE_VIDEO = 'инструкция_огнетушители.mp4' 
PROJECT_DIR = 'runs/detect'
OUTPUT_DIR_NAME = 'video_stream_fps_analysis' 
# Создаем полный путь для сохранения результатов
FULL_OUTPUT_DIR = os.path.join(PROJECT_DIR, OUTPUT_DIR_NAME)


# 1. Загрузка обученной модели
try:
    model = YOLO(MODEL_PATH)
    print(f"✅ Модель загружена: {MODEL_PATH}")
except FileNotFoundError:
    print(f"❌ Ошибка: Файл модели не найден по пути {MODEL_PATH}.")
    exit()


# 2. Запуск детекции в потоковом режиме (stream=True)
print(f"\n--- Запуск детекции и анализ FPS на '{SOURCE_VIDEO}' ---")

# Настройки детекции
conf_threshold = 0.5
iou_threshold = 0.5
device_used = 'mps'  # 'mps', 'cuda', или 'cpu'

results_generator = model.predict(
    source=SOURCE_VIDEO, 
    conf=conf_threshold,     
    iou=iou_threshold,       
    device=device_used,  
    save=True,      
    project=PROJECT_DIR,
    name=OUTPUT_DIR_NAME,
    stream=True            
)

# Инициализация счетчиков для расчета FPS
start_time_total = time.time()
frame_count = 0
fps_history = []


# 3. Итерация генератора и измерение FPS
for result in results_generator:
    
    # Измеряем время для текущего кадра
    frame_end_time = time.time()
    
    # 3.1. Расчет FPS для каждого кадра
    # Обработка первого кадра (для чистого старта)
    if frame_count == 0:
        frame_start_time = frame_end_time 
    else:
        # Время, затраченное на один кадр
        time_per_frame = frame_end_time - frame_start_time
        # FPS для этого кадра
        current_fps = 1.0 / time_per_frame 
        fps_history.append(current_fps)
        
    frame_start_time = frame_end_time # Обновляем старт для следующего кадра
    frame_count += 1


# 4. Финализация и вывод результатов FPS
end_time_total = time.time()
total_time = end_time_total - start_time_total

# Проверка, обработано ли видео
if frame_count > 0:
    # 4.1. Расчет среднего FPS (наиболее точная метрика)
    # Исключаем первый кадр, так как он часто медленнее из-за инициализации
    if fps_history:
        average_fps = sum(fps_history) / len(fps_history)
    else:
        # Если только один кадр, используем общее время
        average_fps = (frame_count - 1) / total_time 

    print("\n--- Результаты анализа ---")
    print(f"Обработано кадров: {frame_count}")
    print(f"Общее время обработки: {total_time:.2f} сек.")
    print(f"✅ Средний FPS (Кадры в секунду): {average_fps:.2f}")

    # 4.2. Путь к результатам
    final_video_path = os.path.join(FULL_OUTPUT_DIR, SOURCE_VIDEO)
    print(f"Результирующее видео сохранено: {final_video_path}")
else:
    print("\n❌ Ошибка: Не удалось обработать ни одного кадра.")