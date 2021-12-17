run_test:
	python3 -m unittest discover -p "*_test.py"

lint:
	pylint ./bigtwo \
	pylint ./playingcards

tensorboard:
	tensorboard --logdir gamerunner/experiments
