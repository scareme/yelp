Restaurant Reviews Generator And String Autocompletion
======================================================
Restaurant reviews generator and string autocompletion model are trained on the dataset available at [https://www.yelp.com/dataset/download](https://www.yelp.com/dataset/download). There are two versions of each model: one trained on bad (1-2 stars) reviews, another trained on good
(5 stars) reviews.

How to run inside docker
------------------------
0. Preparation
- `$ git clone https://github.com/scareme/yelp.git`
- Extract [Yelp Dataset](https://www.yelp.com/dataset/download) to `yelp/yelp_dataset`
- `$ cd yelp`

1. CPU only

- Build docker image:
```bash
$ docker build -t yelp_cpu -f CPUDockerfile .
```

- Run docker container:
```bash
$ docker run --name yelpc --rm -v /absolute/path/to/local/repo/yelp:/yelp -i -t yelp_cpu bash
```

2. (or) with GPU support (You should have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker))

- Build docker image:
```bash
docker build -t yelp_gpu -f GPUDockerfile .
```

- Run docker container:
```bash
docker run --gpus all --name yelpc --rm -v /absolute/path/to/local/repo/yelp:/yelp -i -t yelp_gpu bash
```

How to use
----------

1. `src/preprocess_data.py`: Return bad (bad_review.json) and good (good_review.json) restaurant reviews for further training.

```bash
$ python src/preprocess_data.py --business-file=yelp_dataset/yelp_academic_dataset_business.json --reviews-file=yelp_dataset/yelp_academic_dataset_review.json --output-dir=data
```

2. `src/train_model.py`: Train models.
```bash
$ python src/train_model.py --reviews-path=data/bad_review.json --path-to-save-model=data/bad_model.pt --path-to-save-tokenizer=data/bad_tokenizer.dump
$ python src/train_model.py --reviews-path=data/good_review.json --path-to-save-model=data/good_model.pt --path-to-save-tokenizer=data/good_tokenizer.dump
```

3. `src/generator.py generate`: Generate bad or good reviews starting with --seed-phrase. Depends on [temperature](https://cs.stackexchange.com/questions/79241/what-is-temperature-in-lstm-and-neural-networks-generally) parameter.
```bash
$ python src/generator.py generate --model-path=data/bad_model.pt --tokenizer-path=data/bad_tokenizer.dump --seed-phrase="this was "
====0.2====
this was the worst place to eat in the area. the food was so so so so bad i was not impressed with the food. the chicken was dry and the chicken was not fresh. the sauce was so so. waiter was nice but the food was not good.
==========
$ python src/generator.py generate --model-path=data/good_model.pt --tokenizer-path=data/good_tokenizer.dump --seed-phrase="this was "
====0.2====
this was a great place to eat and drinks. the food was great and the service was excellent. we will definitely be back.
==========
```

4. `src/generator.py autocompletion`: String autocompletion starting with --seed-phrase. Returns possible options in order of log-likelihood descending.
```bash
$ python src/generator.py autocompletion --model-path=data/bad_model.pt --tokenizer-path=data/bad_tokenizer.dump --seed-phrase="the food was " --beam-size=5
['the food was not ', 'the food was good ', 'the food was terrible.', 'the food was good,', 'the food was terrible ']
$ python src/generator.py autocompletion --model-path=data/good_model.pt --tokenizer-path=data/good_tokenizer.dump --seed-phrase="the food was "--beam-size=5
['the food was delicious ', 'the food was amazing ', 'the food was delicious.', 'the food was amazing!', 'the food was excellent ']
```
