# GPU inference image. For air-gapped use, bake the base model + adapter into the image
# (or mount them) and keep HF_HUB_OFFLINE=1 so nothing is fetched at runtime.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY src ./src
COPY configs ./configs
COPY outputs/adapter ./outputs/adapter
ENV PYTHONPATH=/app/src HF_HUB_OFFLINE=1
EXPOSE 8000
CMD ["uvicorn", "qlora_pipeline.serve:app", "--host", "0.0.0.0", "--port", "8000"]
