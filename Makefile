.PHONY: data train eval test serve docker
data:   ; python -m qlora_pipeline.data
train:  ; python -m qlora_pipeline.train
eval:   ; python -m qlora_pipeline.evaluate
test:   ; pytest -q
serve:  ; uvicorn qlora_pipeline.serve:app --host 0.0.0.0 --port 8000
docker: ; docker build -t qlora-extractor .
