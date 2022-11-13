run:
	./run.sh

clean:
	rm -rf data
	rm -rf log_resnet20
	rm -rf save_resnet20
	rm -rf __pycache__

.PHONY: run clean
