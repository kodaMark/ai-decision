FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
COPY . .
EXPOSE 80
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:80", "--workers", "2", "--timeout", "120"]
