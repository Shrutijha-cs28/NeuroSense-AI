# NeuroSenseAI Makefile
.PHONY: setup train-all test app

VENV = .venv
PYTHON = $(VENV)/Scripts/python.exe
PYTEST = $(VENV)/Scripts/pytest.exe
STREAMLIT = $(VENV)/Scripts/streamlit.exe

setup:
	python -m venv $(VENV)
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

train-all:
	$(PYTHON) -m src.alzheimer.train
	$(PYTHON) -m src.parkinsons.train
	$(PYTHON) -m src.parkinsons.train_gait
	$(PYTHON) scripts/generate_synthetic_fusion_data.py
	$(PYTHON) -m src.common.train_fusion --disease alzheimer --data data/synthetic/alzheimer_matched_synthetic.csv
	$(PYTHON) -m src.common.train_fusion --disease parkinsons --data data/synthetic/parkinsons_matched_synthetic.csv

test:
	$(PYTHON) -m pytest tests -v

app:
	$(STREAMLIT) run app.py
