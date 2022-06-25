# Card games

This repository contain codes to train to play card games through reinforcement learning.

### Folder structure

- `algorithm` contains the reinforcement learning algorithms. Currently, only PPO algorithm are implemented and customised for big tow game
- `benchmark` contains logic to pitch different bots against each other to see who is the best
- `bigtwo` contains logic to play the [big two card game](https://en.wikipedia.org/wiki/Big_two)
- `bigtwo_client` contains logic to open up a GUI application to play bigtwo against the bots
- `gamerunner` contains logic to train the bots
- `playingcards` contains logic to represent playing cards

### Installing dependencies

First install [pipenv](https://pipenv.pypa.io/en/latest/), then run `pipenv install`

### Formatter

We used the [black code formatter](https://github.com/psf/black)

### Run BigTwo GUI client

```
cd big_client
PYTHONPATH=.. python client.py
```

### Run BigTwo PPO training

Training configuration is stored in `gamerunner/config``

```
cd gamerunner
PYTHONPATH=.. python ppo_runner.py
```

or if you have [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) installed, you can use docker to run the training

```
docker build -t card-game-runner:$(git rev-parse --short HEAD) .
docker run --rm --gpus all -v $(pwd):/card-games -e PYTHONPATH=.. -it card-game-runner:$(git rev-parse --short HEAD) bash

# cd /card-games/gamerunner
python ppo_runner.py
```

### Run tests

```
make run_test
```