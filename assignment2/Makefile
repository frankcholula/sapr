all: clean extract train eval

.PHONY: extract
extract:
	@python mfcc_extract.py

.PHONY: test
test:
	@pytest -s --log-disable=root

.PHONY: train
train:
	@python train.py

.PHONY: decode
decode:
	@python decoder.py

.PHONY: eval
eval:
	@python eval.py

.PHONY: clean
clean:
	@rm -rf feature_set
	@rm -rf eval_feature_set
	@rm -rf trained_models/hmmlearn
	@rm -rf trained_models/custom
	@rm -rf __pycache__

.PHONY: clean_training
clean_training:
	@rm -rf trained_models/hmmlearn
	@rm -rf trained_models/custom