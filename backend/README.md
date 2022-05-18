# COVID-19 Diagnosis API Server

A high availability Real-time Deep Learning based Diagnosis Model Server for COVID-19 Infection based on Chest X-Ray Radiography. Built on FastAPI, the server handles asynchronous requests and data for Deep Learning model inference.

## Features

- High Availability API

- Docker Containerized

- Easy to deploy

- Intuitive Endpoints

- Modular Design for Better Upgradability

## Tech Stack

1) Docker

2) FastAPI

3) Tensorflow v2 GPU

4) PyTorch

5) MMDetection

6) OpenCV

7) SwaggerUI API Docs

8) SQLAlchemy ORM

9) SQLite3

## Instructions

1) Clone the repo and open the terminal on `/backend` directory.

2) Execute `docker build . -t covid19-diagnosis-api-backend` to build the container image.

3) Run the Application using `docker run covid19-diagnosis-api-backend -p 8080:8080` and you can find the application running on `localhost:8080`

4) Find the API Docs (Swagger UI) at `localhost:8080/docs`
