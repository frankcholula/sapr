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


.PHONY: test_emissions
test_emissions:
	@pytest tests/test_emissions.py

.PHONY: debug
debug:
	@python debugging.py

.PHONY: compare
compare:
	@python compare_training.py