ENV_NAME = ptk-dev

.PHONY: dev-setup

dev-setup:
	conda create -n $(ENV_NAME) python=3.12 -y
	conda run -n $(ENV_NAME) pip install -e ".[dev]"
