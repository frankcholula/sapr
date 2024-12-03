all: clean extract train

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
	@rm -rf trained_models

.PHONY: hmmlearn
hmmlearn:
	@python hmmlearn_model.py