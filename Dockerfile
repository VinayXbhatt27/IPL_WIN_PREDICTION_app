FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
COPY app.py .
COPY ra_pipe.pkl .

RUN pip install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"] 