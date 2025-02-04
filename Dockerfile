FROM python:3.9-slim
WORKDIR /app
COPY ./requirements.txt /app
RUN pip install -r requirements.txt
COPY . .
ENV FLASK_APP=app.py
EXPOSE 5000
VOLUME ["/app/results"]
ENV PYTHONUNBUFFERED=1
CMD ["flask", "run" , "--host", "0.0.0.0", "--port", "5000"]
