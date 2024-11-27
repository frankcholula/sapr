all: extract train

.PHONY: extract
extract:
	@python mfcc_extract.py

.PHONY: test
test:
	@pytest -s --log-disable=root

.PHONY: train
train:
	@python train_hmm.py

.PHONY: clean
clean:
	@rm -rf feature_set
