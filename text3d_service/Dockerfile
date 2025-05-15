FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

ENV FLASK_APP=main.py

EXPOSE 5002

CMD ["flask", "run", "--host=0.0.0.0", "--port=5002"]
