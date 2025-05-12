from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.dummy_operator import DummyOperator
import os

# Пути к файлам
RAW_DATA_PATH = '/tmp/cars_trends_raw.csv'
CLEAN_DATA_PATH = '/tmp/cars_trends_clean.csv'
MODEL_PATH = '/tmp/sgd_cars_model.pkl'

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def download_data(**kwargs):
    """Загрузка данных с GitHub"""
    from dags.train_model import download_raw_data
    return download_raw_data(RAW_DATA_PATH)

def clean_data(**kwargs):
    """Очистка и предобработка данных"""
    from dags.train_model import clean_raw_data
    ti = kwargs['ti']
    input_path = ti.xcom_pull(task_ids='download_data')
    return clean_raw_data(input_path, CLEAN_DATA_PATH)

def prepare_features(**kwargs):
    """Подготовка признаков и масштабирование"""
    from dags.train_model import prepare_features_data
    ti = kwargs['ti']
    input_path = ti.xcom_pull(task_ids='clean_data')
    return prepare_features_data(input_path)

def train_model(**kwargs):
    """Обучение модели с подбором гиперпараметров"""
    from dags.train_model import train_and_log_model
    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='prepare_features')
    return train_and_log_model(data, MODEL_PATH)

def evaluate_model(**kwargs):
    """Оценка модели и сохранение результатов"""
    from dags.train_model import evaluate_trained_model
    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='train_model')
    return evaluate_trained_model(model_path)

with DAG(
    'cars_price_prediction_pipeline',
    default_args=default_args,
    description='Pipeline for predicting car prices based on economic factors',
    schedule_interval=timedelta(days=7),
    catchup=False,
) as dag:
    
    start = DummyOperator(task_id='start')
    
    download_task = PythonOperator(
        task_id='download_data',
        python_callable=download_data,
    )
    
    clean_task = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )
    
    prepare_task = PythonOperator(
        task_id='prepare_features',
        python_callable=prepare_features,
    )
    
    train_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
    )
    
    evaluate_task = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model,
    )
    
    end = DummyOperator(task_id='end')
    
    # Определение порядка выполнения
    start >> download_task >> clean_task >> prepare_task >> train_task >> evaluate_task >> end
