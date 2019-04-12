# DropMax: Adaptive Variational Softmax(Pytorch Implementation)
**DropMax** is a stochastic version of softmax classifier which at each iteration drops non-target classes according to dropout probabilities adaptively decided for each instance. arxiv paper site is [here](https://arxiv.org/abs/1712.07834).


## Usage
This repository is implmented by pytorch, but you can build and run in docker.
You can run experiments using Docker:
```
docker-compose -f docker/docker-compose-cpu.yml build
docker-compose -f docker/docker-compose-cpu.yml run experiment python3 experiment.py
```

## References
H. Beom Lee, J. Lee, S. Kim, E. Yang, S. Ju Hwang, and S. Korea, “DropMax: Adaptive Variational Softmax.”
