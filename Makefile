.PHONY: install run-api run-ui eval test lint download-corpus ingest

install:
	pip install -r requirements.txt

download-corpus:
	python src/download_corpus.py

ingest:
	python -m src.ingest

run-api:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	streamlit run app/streamlit_app.py

eval:
	python -m eval.run_eval
	python -m eval.report

test:
	pytest tests/ -v

lint:
	ruff check src/ app/ eval/ tests/
