# ARS: Adaptive Reward Scaling for Multi-Task Reinforcement Learning submitted in ICML 25

## How to run the code

### Install dependencies

conda create -n icml25ARS python=3.9
conda activate icml25ARS
conda install gcc==12.1.0  or  conda install -c conda-forge libstdcxx-ng=12

```bash
pip install --upgrade pip==21.0
pip install setuptools==65.5.0
pip install wheel==0.41.2

pip install -r requirements.txt

# Installs the Jax library
pip install flax==0.7.5
pip install optax==0.1.7
pip install orbax-checkpoint==0.4.8
pip install chex==0.1.86
pip install --upgrade "jax[cuda12_pip]==0.4.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Run training

export XLA_PYTHON_CLIENT_PREALLOCATE=false && export PYTHONPATH=$(pwd):${PYTHONPATH} && conda activate icml25ARS && export CUDA_VISIBLE_DEVICES=0

MT10
```bash
python run_experiments.py --main_module_name train_ars --seeds 1 2 3 4 --config_args domain=metaworld env_name=MT10 activation=tanh n_layer=4 hidden_dim=400 random_goal=True batch_size=100 n_reset=4 critic_layernorm=True  critic_init_layernorm=True
```

MT10 with LoRA
```bash
python run_experiments.py --main_module_name train_ars --seeds 1 2 3 4 --config_args domain=metaworld env_name=MT10 activation=tanh n_layer=4 hidden_dim=400 random_goal=True batch_size=100 n_reset=4 critic_layernorm=True critic_init_layernorm=True rank=8 threshold=0.8
```

MT50
```bash
python run_experiments.py --main_module_name train_ars --seeds 1 2 3 4 --config_args domain=metaworld env_name=MT50 activation=tanh n_layer=4 hidden_dim=400 random_goal=True batch_size=100 replay_buffer_size=500000 n_reset=6 critic_layernorm=True  critic_init_layernorm=True
```

MT50 with LoRA
```bash
python run_experiments.py --main_module_name train_ars --seeds 1 2 3 4 --config_args domain=metaworld env_name=MT50 activation=tanh n_layer=4 hidden_dim=400 random_goal=True batch_size=100 replay_buffer_size=500000 n_reset=6 critic_layernorm=True  critic_init_layernorm=True rank=16 threshold=0.65
```


## Misc
The implementation is based on [JAXRL](https://github.com/ikostrikov/jaxrl).
