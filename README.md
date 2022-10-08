# Card games

This repository contain codes to train to play card games through reinforcement learning.

### Folder structure

- `algorithm` contains reinforcement learning algorithms. Currently, only PPO algorithm are implemented and customised for big two.
- `benchmark` pitching different bots against each other to see who is the best
- `bigtwo` code to run the [big two card game](https://en.wikipedia.org/wiki/Big_two)
- `bigtwo_client` a simple GUI application to play bigtwo against the bots
- `gamerunner` training the bots using self implemented PPO algorithm
- `rayrunner`: training the bots using Ray RLlib
- `playingcards`: classes that represent playing cards

### Installing dependencies

First [poetry](https://python-poetry.org/), then run `poetry install`

### Formatter

We use the [black code formatter](https://github.com/psf/black)

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

### Run tests

```
make run_test
```