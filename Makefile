.PHONY: build-data validate-gold

build-data:
	python -m src.data.build_silver
	python -m src.data.build_gold

validate-gold:
	python -m src.data.validate_gold_contracts
