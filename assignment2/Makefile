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
	@rm -rf trained_models
	

.PHONY: decode
decode:
	@python decoder.py

.PHONY: eval
eval:
	@python eval_hmm.py
