.PHONY: help install fetch data model demo run
help:
	@echo "BloomCast targets:"
	@echo "  make install   install python deps"
	@echo "  make fetch     download raw public data (FWC + NDBC)"
	@echo "  make data      build fused daily dataset"
	@echo "  make model     train + evaluate (held-out 2018 AUC)"
	@echo "  make demo      precompute dashboard data"
	@echo "  make run       launch the dashboard locally"
install:
	pip install -r requirements.txt
fetch:
	bash fetch_data.sh
data:
	python3 pipeline/build_dataset.py
model:
	python3 model/risk_model.py
demo:
	python3 model/prepare_demo_data.py
run:
	streamlit run app/streamlit_app.py
