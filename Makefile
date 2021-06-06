run_random:
	PYTHONPATH=$(shell pwd) python3 gamerunner/random_runner.py

run_ppo:
    PYTHONPATH=$(shell pwd) python3 gamerunner/ppo_runner.py

lint:
	pylint ./bigtwo \
	pylint ./playingcards

setup:
	sudo apt-get update \
	sudo apt install python3 \
	sudo apt install python3-pip \
	pip3 install -r requirement.txt

update_package:
	pip install git+ssh://git@github.com/Wal8800/gym-playground.git@2.1.0

tensorboard:
    tensorboard --logdir gamerunner/tensorboard
