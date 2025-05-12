import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import joblib
import os

def download_raw_data(output_path):
    """Загрузка сырых данных с GitHub"""
    df = pd.read_csv(
        'https://raw.githubusercontent.com/denisnaenko/cars_trends_dataset/refs/heads/main/automobile_prices_economics_2019_2023.csv',
        delimiter=","
    )
    df.to_csv(output_path, index=False)
    return output_path

def clean_raw_data(input_path, output_path):
    """Очистка и предобработка данных"""
    df = pd.read_csv(input_path)
    
    # Очистка данных
    df = df.dropna(subset=['Month/Year'])
    df['New Price ($)'] = df['New Price ($)'].str.replace(',', '').astype(float)
    df['Used Price ($)'] = df['Used Price ($)'].str.replace(',', '').astype(float)
    df['Inflation Rate (%)'] = df['Inflation Rate (%)'].str.replace('%', '').astype(float)
    df['Interest Rate (%)'] = df['Interest Rate (%)'].str.replace('%', '').astype(float)
    
    df['Units Sold'] = df['Units Sold'].str.replace(',', '')
    df['Units Sold'] = df['Units Sold'].fillna(0).astype(int)

    # Преобразование даты
    df['Year'] = df['Month/Year'].apply(lambda x: int('20' + x.split('-')[0]))
    df['Month'] = df['Month/Year'].apply(lambda x: x.split('-')[1])
    df.drop(columns=['Month/Year'], inplace=True)

    # Удаление выбросов
    df = df[df['New Price ($)'] > 1000]
    df = df[df['Used Price ($)'] > 500]
    df = df[df['Inflation Rate (%)'] < 50]
    df = df[df['Interest Rate (%)'] < 50]
    df = df[df['Units Sold'] > 0]

    df = df.reset_index(drop=True)
    df.to_csv(output_path, index=False)
    
    return output_path

def prepare_features_data(input_path):
    """Подготовка признаков и масштабирование"""
    df = pd.read_csv(input_path)
    
    # Преобразование месяца в числовой формат
    month_map = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4,
        'May': 5, 'Jun': 6, 'Jul': 7, 'Aug': 8,
        'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    df['Month'] = df['Month'].map(month_map)
    
    # Разделение на признаки и целевую переменную
    X = df.drop(columns=['New Price ($)'])
    y = df['New Price ($)']
    
    # Масштабирование
    scaler = StandardScaler()
    power_trans = PowerTransformer()
    X_scaled = scaler.fit_transform(X)
    y_scaled = power_trans.fit_transform(y.values.reshape(-1, 1))
    
    # Сохранение преобразователей
    joblib.dump(scaler, '/tmp/feature_scaler.pkl')
    joblib.dump(power_trans, '/tmp/target_transformer.pkl')
    
    return {
        'X': X_scaled.tolist(),
        'y': y_scaled.tolist(),
        'feature_names': list(X.columns)
    }

def train_and_log_model(data, model_path):
    """Обучение модели с подбором гиперпараметров и логированием"""
    # Восстановление данных
    X = np.array(data['X'])
    y = np.array(data['y'])
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Параметры для GridSearch
    params = {
        'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
        'l1_ratio': [0.001, 0.05, 0.01, 0.2],
        'penalty': ["l1", "l2", "elasticnet"],
        'loss': ['squared_error', 'huber', 'epsilon_insensitive'],
        'fit_intercept': [False, True],
    }
    
    # Настройка MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("cars_price_prediction")
    
    with mlflow.start_run():
        # Обучение с подбором параметров
        lr = SGDRegressor(random_state=42)
        clf = GridSearchCV(lr, params, cv=3, n_jobs=4)
        clf.fit(X_train, y_train.ravel())
        best_model = clf.best_estimator_
        
        # Загрузка преобразователя для целевой переменной
        power_trans = joblib.load('/tmp/target_transformer.pkl')
        
        # Предсказания и обратное преобразование
        y_pred = best_model.predict(X_val)
        y_price_pred = power_trans.inverse_transform(y_pred.reshape(-1, 1))
        y_price_val = power_trans.inverse_transform(y_val)
        
        # Расчет метрик
        rmse = np.sqrt(mean_squared_error(y_price_val, y_price_pred))
        mae = mean_absolute_error(y_price_val, y_price_pred)
        r2 = r2_score(y_price_val, y_price_pred)
        
        # Логирование в MLflow
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        })
        
        # Логирование модели и артефактов
        mlflow.sklearn.log_model(best_model, "model")
        mlflow.log_artifact('/tmp/feature_scaler.pkl')
        mlflow.log_artifact('/tmp/target_transformer.pkl')
        
        # Сохранение модели локально
        joblib.dump(best_model, model_path)
    
    return model_path

def evaluate_trained_model(model_path):
    """Оценка модели и генерация отчетов"""
    # Здесь можно добавить дополнительную оценку модели
    # или генерацию отчетов, например:
    
    print(f"Model successfully saved at: {model_path}")
    print("Evaluation completed.")
    
    # Пример генерации простого отчета
    report = {
        "model_path": model_path,
        "metrics": {
            "rmse": "See MLflow for detailed metrics",
            "r2_score": "See MLflow for detailed metrics"
        }
    }
    
    # Сохранение отчета
    with open('/tmp/model_report.json', 'w') as f:
        import json
        json.dump(report, f)
    
    return True
