.PHONY: install ingest serve eval eval-full test reset-db

install:
	pip install -e .

ingest:
	bash scripts/ingest_all.sh

serve:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	python eval/run_eval.py

eval-full:
	python eval/run_eval.py --ragas

test:
	pytest -m "not requires_llm" -v

reset-db:
	rm -rf data/chroma/* && touch data/chroma/.gitkeep
