# Lab 3
In this Lab I updated the airflow tasks to run the data pipeline for a Random Forest classifier on the Iris dataset.

## Pipeline Overview

The DAG (`Airflow_Lab1`) runs four tasks in sequence:

1. **load_data** — Loads the Iris dataset from scikit-learn (150 samples, 3 classes)
2. **data_preprocessing** — Applies MinMax scaling and splits into 80/20 train/test sets
3. **build_save_model** — Trains a Random Forest classifier (100 estimators) and saves the model
4. **load_model** — Loads the saved model, evaluates on the test set, and prints accuracy and a classification report

## Prerequisites

- Docker and Docker Compose installed
- Ports 8080 available on your machine

## Running the Lab

1. Navigate to the Lab3/Lab_1 directory
2. Run `docker compose up -d`
3. Wait ~60 seconds for the services to become healthy
3. View the airflow webserver at http://localhost:8080
4. Login with username: airflow2 password: airflow2
4. Click the Airflow_Lab1 DAG and trigger the pipeline
4. Click the completed run's load_model_task task, then logs to view the results.

## Stopping the Lab

```bash
docker compose down
```

To also remove the database volume (for a clean restart):

```bash
docker compose down -v
```