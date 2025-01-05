from PIL import Image
import cv2
import numpy as np
import os

# Отримання шляху до робочого столу
desktop_dir = os.path.expanduser("~/Desktop")

# Базова директорія проекту project3 (коренева папка, над .venv)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Папка для збереження результатів обробки (створується в project3/results)
results_folder = os.path.join(base_dir, "results")
os.makedirs(results_folder, exist_ok=True)

# Шляхи до фотографій на робочому столі
photo_paths = [
    os.path.join(desktop_dir, "1.jpg"),
    os.path.join(desktop_dir, "2.jpg"),
    os.path.join(desktop_dir, "3.jpg"),
    os.path.join(desktop_dir, "4.jpg"),
]


def convert_to_grayscale(image_path):
    """Конвертує зображення в ч/б без сторонніх бібліотек."""
    img = Image.open(image_path)
    gray_image = img.convert("L")  # Конвертація в відтінки сірого (Grayscale)
    return np.array(gray_image)


def apply_otsu_threshold(image):
    """Застосування методу Отсу для бінаризації зображення."""
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary_image


def extract_object_using_mask(original_path, binary_mask):
    """Вирізає об'єкт із зображення за допомогою бінарної маски."""
    original_image = cv2.imread(original_path)
    # Застосовуємо маску до зображення
    masked_image = cv2.bitwise_and(original_image, original_image, mask=binary_mask)
    return masked_image


for i, photo_path in enumerate(photo_paths, start=1):
    if not os.path.exists(photo_path):
        print(f"Фото {photo_path} не знайдено на робочому столі. Пропускаємо...")
        continue

    print(f"Обробка зображення: {photo_path}")

    # 1. Конвертація в ч/б
    grayscale_image = convert_to_grayscale(photo_path)
    grayscale_path = os.path.join(results_folder, f"{i}_grayscale.jpg")
    Image.fromarray(grayscale_image).save(grayscale_path)
    print(f"Збережено ч/б зображення: {grayscale_path}")

    # 2. Якщо зображення 3 та 4, виконуємо бінаризацію
    if i in [3, 4]:
        binary_mask = apply_otsu_threshold(grayscale_image)
        binary_path = os.path.join(results_folder, f"{i}_binary.jpg")
        cv2.imwrite(binary_path, binary_mask)
        print(f"Збережено бінарну маску: {binary_path}")

        # 3. Вирізаємо об'єкт за цією маскою
        extracted_object = extract_object_using_mask(photo_path, binary_mask)
        extracted_path = os.path.join(results_folder, f"{i}_extracted.jpg")
        cv2.imwrite(extracted_path, extracted_object)
        print(f"Збережено вирізаний об'єкт: {extracted_path}")
