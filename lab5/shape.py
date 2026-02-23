import os
import pathlib
import dataclasses
import cv2
import numpy
import sklearn.cluster
import collections
import matplotlib.pyplot

# Налаштування шляхів згідно з вимогами
SCRIPT_PATH = pathlib.Path(os.path.abspath(__file__))
SCRIPT_DIR = SCRIPT_PATH.parent
BUILD_DIR = SCRIPT_DIR / 'build'

@dataclasses.dataclass
class CandyObject:
    contour: numpy.ndarray
    center: tuple[int, int]
    mean_bgr: tuple[float, float, float]
    shape_name: str
    color_cluster_id: int = -1
    color_name: str = "Unknown"

def setup_environment() -> None:
    """Створює директорію build, якщо її не існує."""
    if not BUILD_DIR.exists():
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

def determine_shape(contour: numpy.ndarray) -> str:
    """
    Визначає форму об'єкта на основі апроксимації контуру.
    """
    perimeter = cv2.arcLength(contour, True)
    # Апроксимація багатокутником
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)
    
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # Для більш точної перевірки можна додати аналіз співвідношення сторін, 
        # але для наших кубиків вистачить кількості вершин
        return "Square"
    else:
        # Все, що має більше вершин, вважаємо сферою/колом
        return "Circle"

def assign_color_names_to_clusters(kmeans_model: sklearn.cluster.KMeans) -> dict[int, str]:
    """
    Конвертує центри кластерів BGR у HSV для визначення людської назви кольору.
    """
    cluster_names: dict[int, str] = {}
    centers = kmeans_model.cluster_centers_
    
    for i, bgr in enumerate(centers):
        # Перетворюємо центр кластеру у формат HSV для зручної класифікації
        uint8_bgr = numpy.uint8([[bgr]])
        hsv = cv2.cvtColor(uint8_bgr, cv2.COLOR_BGR2HSV)[0][0]
        hue = hsv[0]
        
        # Наближені діапазони відтінків (Hue) в OpenCV (0-179)
        if (0 <= hue <= 15) or (165 <= hue <= 179):
            cluster_names[i] = "Red"
        elif 16 <= hue <= 40:
            cluster_names[i] = "Yellow"
        elif 41 <= hue <= 85:
            cluster_names[i] = "Green"
        elif 86 <= hue <= 140:
            cluster_names[i] = "Blue"
        else:
            cluster_names[i] = "Unknown"
            
    return cluster_names

def process_image(image_path: pathlib.Path) -> tuple[numpy.ndarray, list[CandyObject]]:
    """
    Виконує попередню обробку (Req 3), виділяє об'єкти та кластеризує їх за кольором (Req 2).
    """
    if not image_path.exists():
        print(f"Помилка: Зображення {image_path} не знайдено.")
        return numpy.array([]), []

    img = cv2.imread(str(image_path))
    original_img = img.copy()
    
    # Попередня обробка: розмиття та виділення контурів (Canny)
    blurred = cv2.GaussianBlur(img, (7, 7), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    # Морфологічне закриття для з'єднання розірваних контурів
    kernel = numpy.ones((5, 5), numpy.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candies: list[CandyObject] = []
    
    for contour in contours:
        # Відкидаємо занадто малі об'єкти (шум)
        if cv2.contourArea(contour) < 500:
            continue
            
        # Знаходимо центр об'єкта через моменти
        M = cv2.moments(contour)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        # Визначаємо середній колір об'єкта
        mask = numpy.zeros(gray.shape, dtype=numpy.uint8)
        cv2.drawContours(mask, [contour], -1, 255, -1)
        mean_val = cv2.mean(img, mask=mask)
        mean_bgr = (mean_val[0], mean_val[1], mean_val[2])
        
        shape_name = determine_shape(contour)
        
        candies.append(CandyObject(
            contour=contour,
            center=(cx, cy),
            mean_bgr=mean_bgr,
            shape_name=shape_name
        ))

    # Використання Machine Learning (K-Means) для кластеризації за кольором (Req 2)
    if candies:
        colors_data = numpy.array([candy.mean_bgr for candy in candies])
        # У нас візуально 4 кольори (червоний, синій, зелений, жовтий)
        kmeans = sklearn.cluster.KMeans(n_clusters=4, random_state=42, n_init=10)
        kmeans.fit(colors_data)
        
        cluster_names = assign_color_names_to_clusters(kmeans)
        
        for i, candy in enumerate(candies):
            candy.color_cluster_id = int(kmeans.labels_[i])
            candy.color_name = cluster_names[candy.color_cluster_id]
            
            # Малювання анотацій на зображенні для візуального підтвердження
            cv2.drawContours(original_img, [candy.contour], -1, (0, 255, 0), 2)
            label = f"{candy.color_name} {candy.shape_name}"
            cv2.putText(original_img, label, (candy.center[0] - 40, candy.center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            cv2.putText(original_img, label, (candy.center[0] - 40, candy.center[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return original_img, candies

def aggregate_stats(candies: list[CandyObject]) -> dict[tuple[str, str], int]:
    """Групує цукерки за (Колір, Форма) та підраховує їх."""
    stats: dict[tuple[str, str], int] = collections.defaultdict(int)
    for candy in candies:
        stats[(candy.color_name, candy.shape_name)] += 1
    return dict(stats)

def generate_comparison_report(stats0: dict[tuple[str, str], int], stats1: dict[tuple[str, str], int]) -> str:
    """Генерує текстовий звіт порівняння двох зображень (Req 4)."""
    all_keys = set(stats0.keys()).union(set(stats1.keys()))
    
    report_lines: list[str] = []
    report_lines.append("=== Порівняльний аналіз зображень (Req 4) ===\n")
    report_lines.append(f"{'Об`єкт (Колір, Форма)':<30} | {'Зображення 0':<12} | {'Зображення 1':<12} | {'Різниця':<10}")
    report_lines.append("-" * 70)
    
    total0 = sum(stats0.values())
    total1 = sum(stats1.values())
    
    for key in sorted(all_keys):
        count0 = stats0.get(key, 0)
        count1 = stats1.get(key, 0)
        diff = count1 - count0
        diff_str = f"+{diff}" if diff > 0 else str(diff)
        
        key_str = f"{key[0]} {key[1]}"
        report_lines.append(f"{key_str:<30} | {count0:<12} | {count1:<12} | {diff_str:<10}")
        
    report_lines.append("-" * 70)
    report_lines.append(f"{'ВСЬОГО ОБ`ЄКТІВ (Req 3)':<30} | {total0:<12} | {total1:<12} | {total1 - total0:<10}")
    
    return "\n".join(report_lines)

def main() -> None:
    setup_environment()
    
    img0_path = SCRIPT_DIR / "image_0.png"
    img1_path = SCRIPT_DIR / "image_1.png"
    
    print("Обробка Зображення 0...")
    annotated_img0, candies0 = process_image(img0_path)
    if annotated_img0.size > 0:
        cv2.imwrite(str(BUILD_DIR / "annotated_image_0.png"), annotated_img0)
        
    print("Обробка Зображення 1...")
    annotated_img1, candies1 = process_image(img1_path)
    if annotated_img1.size > 0:
        cv2.imwrite(str(BUILD_DIR / "annotated_image_1.png"), annotated_img1)
        
    stats0 = aggregate_stats(candies0)
    stats1 = aggregate_stats(candies1)
    
    report = generate_comparison_report(stats0, stats1)
    print("\n" + report)
    
    # Зберігаємо звіт
    report_path = BUILD_DIR / "comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[Успіх] Результати та анотовані зображення збережено у папку: {BUILD_DIR}")

if __name__ == "__main__":
    main()