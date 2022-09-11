tests:
	poetry run pytest

update-dependencies:
	# (Re-)lock (= update) dependencies in `poetry.lock` file.
	poetry lock

install-dependencies:
	# Install dependencies, including DEV dependencies, from `poetry.lock` file.
	poetry install --no-root

lint:
	poetry run black . --check
	poetry run isort . --check
	poetry run pylint **/*.py --recursive=true

tensorboard:
	tensorboard --logdir gamerunner/experiments

build_gamerunner:
	docker build -t card-game-runner:$(git rev-parse --short HEAD) .

