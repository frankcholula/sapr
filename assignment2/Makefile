all: extract init

.PHONY: extract
extract:
	python mfcc_extract.py

.PHONY: init
init:
	python initialise_hmm.py	

.PHONY: test
test:
	pytest -s --log-disable=root
