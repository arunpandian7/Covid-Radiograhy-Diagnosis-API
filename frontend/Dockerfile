FROM python:3.10.5-slim-buster

RUN pip install streamlit

WORKDIR /app

COPY . .

CMD ["streamlit", "run", "main.py"]
