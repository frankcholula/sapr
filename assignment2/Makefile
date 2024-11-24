all: extract

.PHONY: extract
extract:
	python mfcc_extract.py

.PHONY: test
test:
	python -m unittest tests/test_mfcc_extract.py