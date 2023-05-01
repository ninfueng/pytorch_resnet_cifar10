MODEL:=resnet20

.PHONY: run
run:
	python -u trainer.py --arch=$(MODEL) --save-dir=save_$(MODEL) | tee -a log_$(MODEL)

.PHONY: fmt
fmt:
	isort .
	black .

.PHONY: clean
clean:
	rm -rf data
	rm -rf log_resnet20
	rm -rf save_resnet20
	rm -rf __pycache__
