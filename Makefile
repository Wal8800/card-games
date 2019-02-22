run_random:
	PYTHONPATH=$(pwd) python3 gamerunner/random_runner.py

run_maddpg:
	PYTHONPATH=$(pwd) python3 gamerunner/maddpg_runner.py

setup:
	sudo apt-get update \
	sudo apt install python3 \
	sudo apt install python-pip \
	pip install -r requirement.txt
