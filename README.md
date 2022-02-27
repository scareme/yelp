1. Run docker
a. CPU
Build docker image:
docker build -t yelp_cpu -f CPUDockerfile .

Run docker container:
docker run --name yelpc --rm -v ~/Projects/yelp:/yelp -i -t yelp_cpu bash

b. GPU (You should have NVIDIA Container Toolkit [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker])
Build docker image:
docker build -t yelp_gpu -f GPUDockerfile .

Run docker container:
docker run --gpus all --name yelpc --rm -v ~/Projects/yelp:/yelp -i -t yelp_gpu bash

2. Preprocess data:
python src/preprocess_data.py --business-file=yelp_dataset/yelp_academic_dataset_business.json --reviews-file=yelp_dataset/yelp_academic_dataset_review.json --output-dir=data


3. Train models:
python src/train_model.py --reviews-path=data/bad_review.json --path-to-save-model=data/bad_model.pt --path-to-save-tokenizer=data/bad_tokenizer.dump
python src/train_model.py --reviews-path=data/good_review.json --path-to-save-model=data/good_model.pt --path-to-save-tokenizer=data/good_tokenizer.dump


4. Use models:
python src/generator.py generate --model-path=data/bad_model.pt --tokenizer-path=data/bad_tokenizer.dump --seed-phrase="the food was "
python src/generator.py generate --model-path=data/good_model.pt --tokenizer-path=data/good_tokenizer.dump --seed-phrase="the food was "
python src/generator.py autocomplete --model-path=data/bad_model.pt --tokenizer-path=data/bad_tokenizer.dump --seed-phrase="the food was " --beam-size=5
python src/generator.py autocomplete --model-path=data/good_model.pt --tokenizer-path=data/good_tokenizer.dump --seed-phrase="the food was " --beam-size=5
