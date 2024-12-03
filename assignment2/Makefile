all: clean extract train eval

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
	@rm -rf eval_feature_set
	@rm -rf trained_models/hmmlearn
	@rm -rf trained_models/custom

.PHONY: clean_training
clean_training:
	@rm -rf trained_models/hmmlearn
	@rm -rf trained_models/custom

.PHONY: decode
decode:
	@python decoder.py

.PHONY: eval
eval:
	@python eval_hmm.py
