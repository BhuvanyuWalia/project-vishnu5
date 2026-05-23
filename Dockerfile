FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY indian_ocean_200x200.json .
COPY Vishnu5_NW_deploy.pth .
COPY Vishnu5_NE_deploy.pth .
COPY Vishnu5_SW_deploy.pth .
COPY Vishnu5_SE_deploy.pth .
COPY static/ ./static/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]