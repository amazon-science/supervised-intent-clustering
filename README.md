# Supervised Intent Clustering
This is a package to fine-tune language models in order to create clustering-friendly embeddings.
It is based on the paper [Supervised clustering loss for clustering-friendly sentence embeddings: An application to intent clustering, (2023) G Barnabo, A Uva, S Pollastrini, C Rubagotti, D Bernardi](https://www.amazon.science/publications/supervised-clustering-loss-for-clustering-friendly-sentence-embeddings-an-application-to-intent-clustering)

## How to cite us
```
@Inproceedings{Barnabo2023,
 author = {Giorgio Barnabo and Antonio Uva and Sandro Pollastrini and Chiara Rubagotti and Davide Bernardi},
 title = {Supervised clustering loss for clustering-friendly sentence embeddings: An application to intent clustering},
 year = {2023},
 url = {https://www.amazon.science/publications/supervised-clustering-loss-for-clustering-friendly-sentence-embeddings-an-application-to-intent-clustering},
 booktitle = {IJCNLP-AACL 2023},
}
```


## Repository Structure
```
p-lightning-template
| conf                      # contains Hydra config files
  | data
  | model
  | train
  root.yaml                 # hydra root config file
| data                      # datasets should go here
| experiments               # where the models are stored
| src
  | pl_data_modules.py      # base LightinigDataModule
  | pl_modules.py           # base LightningModule
  | train.py                # main script for training the network
| README.md
| requirements.txt
| setup.sh                  # environment setup script 
```
The structure of the repository is very simplistic and involves mainly four
components:
- **pl_data_modules.py** where you can declare your LightningDataModules.
- **pl_modules.py** where you can declare your LightningModules.
- **train.py** the main script to start the training phase.
- **conf** the directory containing the config files of Hydra.

## Initializing the environment
In order to set up the python interpreter we utilize [conda](https://docs.conda.io/projects/conda/en/latest/index.html)
, the script `setup.sh` creates a conda environment and install pytorch
and the dependencies in "requirements.txt".


## Using the repository
To use this repository as a starting template for your projects, you can just click the green button "Use this template" at the top of this page. More on using GitHub repositories on the following [link](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template).


## FAQ
**Q**: When I run any script using a Hydra config I can see that relative paths do not work. Why?

**A**: Whenever you run a script that uses a Hydra config, Hydra will create a new working directory
(specified in the root.yaml file) for you. Every relative path you will use will start from it, and this is why you 
get the 'FileNotFound' error. However, using a different working directory for each of your experiments has a couple of 
benefits that you can read in the 
[Hydra documentation](https://hydra.cc/docs/tutorials/basic/running_your_app/working_directory/) for the Working 
directory. There are several ways that hydra offers as a workaround for this problem here we will report the two that
the authors of this repository use the most, but you can find the other on the link we previously mentioned:

1. You can use the 'hydra.utils.to_absolute_path' function that will convert every relative path starting from your 
working directory (p-lightning-template in this project) to a full path that will be accessible from inside the 
new working dir.
   
2. Hydra will provide you with a reference to the original working directory in your config files.
You can access it under the name of 'hydra:runtime.cwd'. So, for example, if your training dataset
has the relative path 'data/train.tsv' you can convert it to a full path by prepending the hydra 
variable before 


## Contributing
Contributions are always more than welcome, the only thing to take into account when submitting a pull request is
that we utilize the [Black](https://github.com/psf/black) code formatter with a max length for the code of 120. 
More pragmatically you should ensure to utilize the command "black -l 120" on the whole "src" directory before pushing
the code. 


## Other useful repositories
This repository has been created with the idea of providing a simple skeleton from which you can 
start a PyTorch Lightning project. Instead of favoring the customizability, we favored the simplicity
and we intended this template as a base for building more specific templates based on the user needs
(for example by forking this one). However, there are several other repositories with different 
features that you can check if interested. We will list two of them here:
- [lucmos/nn-template](https://github.com/lucmos/nn-template): a very nice template with support for
    DVC.
- [hobogalaxy/lightning-hydra-template](https://github.com/hobogalaxy/lightning-hydra-template):
    another useful and very well documented template repository.


## How to use it

### Installation
You need to have conda installed. Please refer to the [conda installation page](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html). The miniconda version is sufficient.

After conda installation, run the following script
```bash
sh setup.sh
```

### Download base models
Download a sentence-transformer model by selecting is name here: [HuggingFace Sentence-Transformer](https://huggingface.co/sentence-transformers).

E.g., suppose that you selected `paraphrase-multilingual-MiniLM-L12-v2`, the run this code.
Run:
```bash
python download_base_model.py <name-of-the-model>
```
It will download it in the folder [base_language_models](./base_language_models/).

when we tested our model we used the following four base sentence encoders base_language_model_folder:
1. `bert-base-multilingual-cased`
2. `xlm-roberta-base`
3. `sentence-transformers_all-mpnet-base-v2`
4. `paraphrase-multilingual-mpnet-base-v2`

We suggest to use `paraphrase-multilingual-mpnet-base-v2`, which gives good performances even without fine-tuning.

### Fine-Tuning
To fine-tune any of the 4 base sentence encoders you should follow these steps.
1. given some utterances that you want to use for fine-tuning, create train, dev and test sets such that each intent is represented in only one of the three splits; utterances can come from different domains but in that case some intents from each domain should be present in each split;
    1. for example, let’s assume that you have utterances with the following intents: "CreateAccount", "ChangeAddress", "ResetPassword", "RemoveDependent", "ReportAutomobileAccident", "AddDependent", "ReportBillingIssue", "UpdatePaymentPreference", "CancelPlan", "RequestProofOfInsurance"
    2. you could first randomly assign 33% of these intents to the train set, 33% to the dev set and 33% to the test set and then place all the utterances with these intents in the corresponding dataset split;
2. upload your dataset splits in the data folder `supervised-intent-clustering/data/New_Fine_Tuning_Dataset`. 
    Each file (train, dev, test) should come in the form of a csv file with the following columns: `dataset,utterance_id,utterance_split,utterance_lang,utterance_text,utterance_intent`
    1. the folder already contains three sample splits that can be open to better understand how to create the necessary datasets
3. modify the *dataset-specific hyper-parameters* in the corresponding file `supervised-intent-clustering/conf/dataset_specific_hyperparams/new_fine_tuning_dataset.yaml`
    1. **language_list**: a list representing a subset of language identifiers that you used in the creation of your datasets; if your datasets contain many languages you can perform training only on a subset of those;
    2. **modality**: a string that you can add to the named of the fine-tuned model; you can leave it blank
    3. **intent_classes_per_batch**: a number between 2 and min(#train_intents, #dev_intents, #test_intents); the greater the better;
    4. **samples_per_class_per_batch**: how many utterances for each intents to add in each batch; ideally the total number of utterances per batch (intent_classes_per_batch*samples_per_class_per_batch) should be around 150; if not enough utterances for an intent are found the available ones are randomly sampled with repetition;
        1. remark: `intent_classes_per_batch*samples_per_class_per_batch` should not be greater than the number of examples in train, dev or test, otherwise the error `IndexError: list index out of range` will be raised.
    5. **limit_train_batches**: number of batches per training epoch - we suggest a small number between 4 and 10; 
    6. **limit_val_batches**:  number of batches used to collect validation metrics - we suggest a small number between 4 and 10; 
4. finally, to launch the fine-tuning you just need to run the following command:
      `PYTHONPATH=. python3 ./src/train.py dataset_specific_hyperparams=new_fine_tuning_dataset model.base_model_name=<the_base_model_you_want_to_fine_tune> model.training_objective=<the_training_objective_you_want_to_use``>`
    1. model.base_model_name: when we tested our model we used the following four base sentence encoders base_language_model_folder:
        1. `bert-base-multilingual-cased`
        2. `xlm-roberta-base`
        3. `sentence-transformers_all-mpnet-base-v2`
        4. `paraphrase-multilingual-mpnet-base-v2`
    2. We suggest using `sentence-transformers_all-mpnet-base-v2` for English-only datasets and `paraphrase-multilingual-mpnet-base-v2` for the multilingual ones
    3. model.training_objective: you can choose among the following losses:
        1. contrastive_learning
        2. cosine_similarity_loss
        3. triplet_margin_loss
        4. supervised_learning
        5. binary_classification
    4. We suggest using either the `supervised_learning` loss or the `triplet_margin_loss`. Default hyper-parameters should work just fine.
5. when the training finishes due to early stopping, you will find your fine-tuned model in the corresponding folder `supervised-intent-clustering/fine_tuned_language_models`. The produce sentence encoder can be directly used with the Hugging Face Sentence Encoder Library:
    1. from sentence_transformers import SentenceTransformer
        
        def get_sentence_embeddings(list_of_sentences: List[str], 
                hf_model_name_or_path: str=<path_to_the_folder_where_the_model_was_saved>):
        
            model = SentenceTransformer(hf_model_name_or_path)
            sentence_embeddings = model.encode(
                list_of_sentences, batch_size=64, show_progress_bar=True)
            return sentence_embeddings
6. For each experiment you run, key performance metrics will be logged in the file `supervised-intent-clustering/experiment_metrics.txt.` In particular, you want the PRAUC after training to be higher then the PRAUC before training on both the train, dev and test sets. If you run multiple experiments, you can turn the `experiment_metrics.txt` file into a more readable .csv file running the following command:
    1. `PYTHONPATH=. python3 ./src/`training_log_post_processing.py
7. To further customise the training behaviour, you can manually modify the following files:
    1. `supervised-intent-clustering/conf/model/default_model.yaml`
    2. `supervised-intent-clustering/conf/train/default_train.yaml`
    3. `supervised-intent-clustering/conf/data/default_data.yaml`
    4. `supervised-intent-clustering/src/train.py`