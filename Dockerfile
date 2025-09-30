FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# We only need to copy our app and the model file.
COPY ./app.py .
COPY ./model.pkl .

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "app:app"]