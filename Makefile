run_random:
	PYTHONPATH=$(shell pwd) python3 gamerunner/random_runner.py

run_maddpg:
	PYTHONPATH=$(shell pwd) python3 gamerunner/maddpg_runner.py

setup:
	sudo apt-get update \
	sudo apt install python3 \
	sudo apt install python3-pip \
	pip3 install -r requirement.txt
