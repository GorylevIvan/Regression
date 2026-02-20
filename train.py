import sys
import json
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# CatBoost — опционально (можно не устанавливать catboost для работы программы)
try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False



# Метрики
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {
        "mse": float(mse),
        "rmse": float(rmse),
        "r2": float(r2)
    }


# Делим в соотношении 60 / 20 / 20
def split_data(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# Ridge
def train_ridge(X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n Тренируем Ridge")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("regressor", Ridge(alpha=1.0))
    ])

    model.fit(X_train, np.log1p(y_train))

    # Validation
    y_val_pred = np.expm1(model.predict(X_val))
    val_metrics = evaluate(y_val, y_val_pred)

    print("\n Ridge | Validation")
    print(f"MSE : {val_metrics['mse']:.2f}")
    print(f"RMSE: {val_metrics['rmse']:.2f}")
    print(f"R2  : {val_metrics['r2']:.3f}")

    # Test
    y_test_pred = np.expm1(model.predict(X_test))
    test_metrics = evaluate(y_test, y_test_pred)

    print("\n Ridge | Test")
    print(f"MSE : {test_metrics['mse']:.2f}")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"R2  : {test_metrics['r2']:.3f}")

    return model, val_metrics, test_metrics



# CatBoost (опционально и вообще просто ради интереса, можно сюда даже не смотреть)
def train_catboost(X_train, y_train, X_val, y_val, X_test, y_test):
    if not CATBOOST_AVAILABLE:
        print("\n CatBoost не установлен — пропускаем")
        return None

    print("\n Тренируем CatBoost")

    model = CatBoostRegressor(
        iterations=500,
        depth=8,
        learning_rate=0.05,
        loss_function="RMSE",
        random_seed=42,
        verbose=False
    )

    model.fit(X_train, y_train)

    # Validation
    y_val_pred = model.predict(X_val)
    val_metrics = evaluate(y_val, y_val_pred)

    print("\n CatBoost | Validation")
    print(f"MSE : {val_metrics['mse']:.2f}")
    print(f"RMSE: {val_metrics['rmse']:.2f}")
    print(f"R2  : {val_metrics['r2']:.3f}")

    # Test
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate(y_test, y_test_pred)

    print("\n CatBoost | Test")
    print(f"MSE : {test_metrics['mse']:.2f}")
    print(f"RMSE: {test_metrics['rmse']:.2f}")
    print(f"R2  : {test_metrics['r2']:.3f}")

    return model



# Main
def main(x_path, y_path):
    X = np.load(x_path)
    y = np.load(y_path)

    print(f" Dataset загружен: X={X.shape}, y={y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print(f"Train: {X_train.shape}")
    print(f"Val  : {X_val.shape}")
    print(f"Test : {X_test.shape}")

   
    # Ridge — основная модель
    ridge_model, val_metrics, test_metrics = train_ridge(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )


    # CatBoost — эксперимент
    train_catboost(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    # Финальный этап обучения (train + val)
    print("\n Финальный этап обучения Ridge на train + val")

    X_final = np.vstack([X_train, X_val])
    y_final = np.concatenate([y_train, y_val])

    ridge_model.fit(X_final, np.log1p(y_final))


    # Сохраняет модель и метрики
    resources = Path("resources")
    resources.mkdir(exist_ok=True)

    model_path = resources / "model.joblib"
    joblib.dump(ridge_model, model_path)

    metrics = {
        "model": "Ridge",
        "validation": val_metrics,
        "test": test_metrics
    }

    metrics_path = resources / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    print(f"\n Ridge модель сохранена в {model_path}")
    print(f" Метрики сохранены в {metrics_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])