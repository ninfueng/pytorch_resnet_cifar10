.PHONY: run
run:
	./run.sh

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
