import sys
import subprocess
import numpy as np
import joblib
import json
from pathlib import Path

# Пути к сохранённым файлам модели и метрик
MODEL_PATH = Path("resources/model.joblib")
METRICS_PATH = Path("resources/metrics.json")


def train_if_needed(x_path, y_path):
    """Проверяет, есть ли уже обученная модель. Если нет - запускает обучение"""
    if MODEL_PATH.exists():
        print("Модель уже есть")
        return

    if y_path is None:
        raise ValueError(
            "Модель не найдена. Для начала добавьте к переменным y_data.npy"
        )

    print(" Модель не найдена. Начинаем обучение...")

    # Запускает отдельный скрипт обучения модели
    subprocess.run(
        ["python", "train.py", x_path, y_path],
        check=True
    )


def print_metrics():
    """Выводит метрики модели из сохранённого JSON-файла"""
    if not METRICS_PATH.exists():
        print("Файл с метриками не найден")
        return

    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    print("\n Метрики:")
    print(f"Model: {metrics.get('model')}")

    # Выводит метрики для валидации и теста, если они есть
    for split in ["validation", "test"]:
        if split in metrics:
            m = metrics[split]
            print(f"\n{split.capitalize()}:")
            print(f"  MSE : {m['mse']:.2f}")
            print(f"  RMSE: {m['rmse']:.2f}")
            print(f"  R2  : {m['r2']:.3f}")


def predict(x_path, y_path=None):
    """Загружает модель и делает предсказания на новых данных"""
    model = joblib.load(MODEL_PATH)
    X = np.load(x_path)

    print(f"\n Грузим X: {X.shape}")

    # Делает предсказания и преобразуем через expm1 (обратное логарифмирование)
    y_pred = np.expm1(model.predict(X)).astype(float)

    if y_path is None:
        print("\n Предсказанные зарплаты:")
        print(y_pred.tolist())
        return

    # Если передан y_true - сравниваем предсказания с истинными значениями
    y_true = np.load(y_path)

    print("\n Предсказанные и реальные зарплаты:")
    print(" idx | predicted | real ")
    print("-" * 32)

    for i in range(min(30, len(y_true))):
        print(
            f"{i:>3} | "
            f"{y_pred[i]:>9.0f} | "
            f"{y_true[i]:>9.0f}"
        )


def main():
    """Основная функция"""
    if len(sys.argv) not in (2, 3):
        print("Usage:")
        print("  python app.py x_data.npy")
        print("  python app.py x_data.npy y_data.npy")
        sys.exit(1)

    x_path = sys.argv[1]
    y_path = sys.argv[2] if len(sys.argv) == 3 else None

    train_if_needed(x_path, y_path)
    print_metrics()
    predict(x_path, y_path)


if __name__ == "__main__":
    main()