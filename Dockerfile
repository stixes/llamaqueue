FROM python:3.11-slim
WORKDIR /app
RUN pip install fastapi "uvicorn[standard]" httpx
COPY proxy.py .
EXPOSE 8000
CMD ["uvicorn", "proxy:app", "--host", "0.0.0.0", "--port", "8000"]
