run_random:
	PYTHONPATH=$(shell pwd) python3 gamerunner/random_runner.py

lint:
    pylint ./bigtwo

setup:
	sudo apt-get update \
	sudo apt install python3 \
	sudo apt install python3-pip \
	pip3 install -r requirement.txt

update_package:
    pip install git+ssh://git@github.com/Wal8800/gym-playground.git@0.2.0