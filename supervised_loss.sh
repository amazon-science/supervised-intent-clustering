PYTHONPATH=. python3 ./src/train.py dataset_specific_hyperparams=new_fine_tuning_dataset model.base_model_name=paraphrase-mpnet-base-v2 model.training_objective=supervised_learning model.C=1.1 model.r=1.0
#PYTHONPATH=. python3 ./src/train.py dataset_specific_hyperparams=new_fine_tuning_dataset model.base_model_name=bert-base-multilingual-cased model.training_objective=supervised_learning model.C=1.0 model.r=1.0
PYTHONPATH=. python3 ./src/train.py dataset_specific_hyperparams=new_fine_tuning_dataset model.base_model_name=paraphrase-multilingual-mpnet-base-v2 model.training_objective=supervised_learning model.C=1.1 model.r=1.0
#PYTHONPATH=. python3 ./src/train.py dataset_specific_hyperparams=new_fine_tuning_dataset model.base_model_name=xlm-roberta-base model.training_objective=supervised_learning model.C=1.0 model.r=1.0
