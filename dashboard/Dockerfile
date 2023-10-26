FROM python:3.10-slim-buster

RUN pip install explainerdashboard
RUN pip install dash==2.6.2

COPY tabs.py ./
COPY dashboard.py ./
COPY app.py ./

RUN python dashboard.py

EXPOSE 9050
CMD ["python", "./app.py"]
