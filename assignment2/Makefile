all: extract

.PHONY: extract
extract:
	python mfcc_extract.py

.PHONY: test
test:
	pytest -s --log-disable=root
