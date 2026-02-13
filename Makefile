.PHONY: build-data validate-gold build-geometries performance-test

build-data:
	python -m src.data.migrate_release_data
	python -m src.data.build_silver
	python -m src.data.build_gold

validate-gold:
	python -m src.data.validate_gold_contracts

build-geometries:
	python scripts/simplify_geometries.py

performance-test:
	python scripts/performance_test.py
