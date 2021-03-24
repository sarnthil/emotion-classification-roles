all: predict

.PHONY: all clean train predict

clean:
	rm -rf workdata/*


### prepare the data for the masking experiment (1)

workdata/unified.json: scripts/unify.py sources/splitted.json
	python3 scripts/unify.py sources/splitted.json >workdata/unified.json

workdata/masking_experiment/datasets/eca/text_emotion_all.tsv.train: scripts/masking_experiment/mask.py workdata/unified.json
	python3 scripts/masking_experiment/mask.py workdata/unified.json 

### generate experiment configs for all experiments within the masking experiment

workdata/masking_experiment/configs: scripts/masking_experiment/generate-experiment-configs.py workdata/masking_experiment/datasets/eca/text_emotion_all.tsv.train
	python3 scripts/masking_experiment/generate-experiment-configs.py

### train models according to the configs

workdata/masking_experiment/models/eca: workdata/masking_experiment/configs scripts/masking_experiment/train-allennlp.py
	python3 scripts/masking_experiment/train-allennlp.py eca

workdata/masking_experiment/models/gne: workdata/masking_experiment/configs scripts/masking_experiment/train-allennlp.py
	python3 scripts/masking_experiment/train-allennlp.py gne

workdata/masking_experiment/models/emotion-stimulus: workdata/masking_experiment/configs scripts/masking_experiment/train-allennlp.py
	python3 scripts/masking_experiment/train-allennlp.py emotion-stimulus

workdata/masking_experiment/models/electoral_tweets: workdata/masking_experiment/configs scripts/masking_experiment/train-allennlp.py
	python3 scripts/masking_experiment/train-allennlp.py electoral_tweets

workdata/masking_experiment/models/reman: workdata/masking_experiment/configs scripts/masking_experiment/train-allennlp.py
	python3 scripts/masking_experiment/train-allennlp.py reman

train: workdata/masking_experiment/models/eca workdata/masking_experiment/models/gne workdata/masking_experiment/models/emotion-stimulus workdata/masking_experiment/models/electoral_tweets workdata/masking_experiment/models/reman

### make predictions based on the trained models

workdata/masking_experiment/datasets/eca/predict_all.json.test: scripts/masking_experiment/make-prediction-jsons.py
	python3 scripts/masking_experiment/make-prediction-jsons.py

workdata/masking_experiment/predictions/eca: scripts/masking_experiment/predict-allennlp.py workdata/masking_experiment/datasets/eca/predict_all.json.test
	python3 scripts/masking_experiment/predict-allennlp.py eca

workdata/masking_experiment/predictions/gne: scripts/masking_experiment/predict-allennlp.py workdata/masking_experiment/datasets/gne/predict_all.json.test
	python3 scripts/masking_experiment/predict-allennlp.py gne

workdata/masking_experiment/predictions/emotion-stimulus: scripts/masking_experiment/predict-allennlp.py workdata/masking_experiment/datasets/emotion-stimulus/predict_all.json.test
	python3 scripts/masking_experiment/predict-allennlp.py emotion-stimulus

workdata/masking_experiment/predictions/electoral_tweets: scripts/masking_experiment/predict-allennlp.py workdata/masking_experiment/datasets/electoral_tweets/predict_all.json.test
	python3 scripts/masking_experiment/predict-allennlp.py electoral_tweets

workdata/masking_experiment/predictions/reman: scripts/masking_experiment/predict-allennlp.py workdata/masking_experiment/datasets/reman/predict_all.json.test
	python3 scripts/masking_experiment/predict-allennlp.py reman

predict: workdata/masking_experiment/predictions/eca workdata/masking_experiment/predictions/gne workdata/masking_experiment/predictions/emotion-stimulus workdata/masking_experiment/predictions/electoral_tweets workdata/masking_experiment/predictions/reman

### evaluate the models

workdata/masking_experiment/predictions/emotion-stimulus/all.aggregated: scripts/masking_experiment/aggregate-predictions-and-gold.py
	python3 scripts/masking_experiment/aggregate-predictions-and-gold.py

workdata/masking_experiment/results.json: scripts/masking_experiment/calculate-fscores-from-aggregations.py
	python3 scripts/masking_experiment/calculate-fscores-from-aggregations.py


