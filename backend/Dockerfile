FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# HuggingFace Spaces exposes port 7860
EXPOSE 7860

# Pre-download ClinicalBERT model during build so runtime is fast
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT'); AutoModel.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')" || true

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
