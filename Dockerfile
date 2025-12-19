FROM python:3.10-slim

WORKDIR /app

# Copy entire project (including venv)
COPY . .

# Copy pre-installed python packages from venv into image
COPY venv/Lib/site-packages /usr/local/lib/python3.10/site-packages

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

