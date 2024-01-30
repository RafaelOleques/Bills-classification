import sys
import wandb
import json
import pandas as pd
import numpy as np
import os.path
from os import path
import os

root = "./files"
sys.path.append(f'{root}/bert_trainer')

from files.bert_trainer.model import Model
from datasets import Dataset, DatasetDict

from legalnlp.clean_functions import clean_bert

class Cross_validation:

    def __init__(self, corpus, bert_list, validation_name, folds=10, random_state=42, input="ementa", target="temas", max_length=512):
        self.corpus = corpus
        self.target = target
        self.input = input
        self.validation_name = validation_name
        print(f"{root}/bills-dataset/{self.validation_name}/{self.input}/{self.corpus}/labels.csv")
        self.labels, self.id2label, self.label2id = self._get_labels(pd.read_csv(f"{root}/bills-dataset/{self.validation_name}/{self.input}/{self.corpus}/labels.csv"))
        self.bert_list = bert_list
        self.folds = folds
        self.random_state = random_state
        self.max_length = max_length

    def _get_labels(self, df_labels):
        #Labels
        labels = df_labels.labels.values.tolist()
        id2label = {idx:label for idx, label in enumerate(labels)}
        label2id = {label:idx for idx, label in enumerate(labels)}

        return labels, id2label, label2id

    def _create_directory(self, ref):
        if path.exists(ref) == False:
            os.mkdir(ref)


    def run(self, evaluation_strategy, eval_steps, save_total_limit, save_strategy,learning_rate,weight_decay, batch_size, num_train_epochs, metric_for_best_model, adam_beta1, adam_beta2, start=0, end=None):
        for bert_name in self.bert_list:
            bert_version = self.bert_list[bert_name]

            if end is None:
                end = self.folds

            for k in range(start, end):
                name = f"{corpus}_{self.input}_{self.max_length}_{bert_name}_fold{k}"
                
                hyperparameters = {
                    "bert_version": self.bert_list[bert_name],

                    #Labels,
                    "labels": self.labels,
                    "id2label": self.id2label,
                    "label2id": self.label2id,

                    #Hiperparameters,
                    "name": name,
                    "evaluation_strategy": evaluation_strategy,
                    "eval_steps": eval_steps,
                    "save_total_limit": save_total_limit,
                    "save_strategy": save_strategy,

                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,

                    "batch_size": batch_size,
                    "num_train_epochs": num_train_epochs,
                    "metric_for_best_model": metric_for_best_model,

                    "adam_beta1": adam_beta1,
                    "adam_beta2":adam_beta2,
                }

                #metrics dict
                metrics = {}

                #Starting model
                model = Model(bert_version=bert_version, labels=self.labels, id2label=self.id2label, label2id=self.label2id, input=self.input, max_length=self.max_length)


                #pandas df
                df_train = pd.read_csv( f"{root}/bills-dataset/{self.validation_name}/{self.input}/{self.corpus}/{self.folds}folds/fold{k}/train.csv")
                df_test = pd.read_csv(f"{root}/bills-dataset/{self.validation_name}/{self.input}/{self.corpus}/{self.folds}folds/fold{k}/test.csv")
                df_validation = pd.read_csv(f"{root}/bills-dataset/{self.validation_name}/{self.input}/{self.corpus}/{self.folds}folds/fold{k}/validation.csv")
                
                if bert_name == "bertikal":
                    #Themes string to list
                    df_train[self.input] = [clean_bert(input) for input in df_train[self.input]]
                    df_test[self.input] = [clean_bert(input) for input in df_test[self.input]]
                    df_validation[self.input] = [clean_bert(input) for input in df_validation[self.input]]

                #Themes string to list
                df_train.temas = df_train.temas.str.split('; ')
                df_test.temas = df_test.temas.str.split('; ')
                df_validation.temas = df_validation.temas.str.split('; ')

                #dataset
                dataset_train = Dataset.from_pandas(df_train)
                dataset_test = Dataset.from_pandas(df_test)
                dataset_validation = Dataset.from_pandas(df_validation)     


                #Training
                model.set_dataset(dataset_train, dataset_validation, dataset_test)
                model.training_arguments(name, evaluation_strategy, eval_steps, save_total_limit, save_strategy, 
                                            learning_rate, weight_decay, 
                                            batch_size, num_train_epochs, metric_for_best_model, 
                                            adam_beta1, adam_beta2)
                model.train()
                
                print(f"======> fold {k}")
                
                print(f"Saving model...")

                #Saving model
                self._create_directory(f"{root}/models")
                self._create_directory(f"{root}/models/{self.validation_name}")
                self._create_directory(f"{root}/models/{self.validation_name}/{self.input}")
                self._create_directory(f"{root}/models/{self.validation_name}/{self.input}/{self.max_length}")
                self._create_directory(f"{root}/models/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}")
                self._create_directory(f"{root}/models/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds")
                self._create_directory(f"{root}/models/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}")
                self._create_directory(f"{root}/models/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}/fold{k}")

                model.save_model(f"{root}/models/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}/fold{k}")
                
                
                #Metrics
                #validation_metrics = model.get_metrics_validation()
                test_probs, test_metrics = model.get_probs_and_metrics_test()

                #Hyperparameters
                self._create_directory(f"{root}/hyperparameters")
                self._create_directory(f"{root}/hyperparameters/{self.validation_name}")
                self._create_directory(f"{root}/hyperparameters/{self.validation_name}/{self.input}")
                self._create_directory(f"{root}/hyperparameters/{self.validation_name}/{self.input}/{self.max_length}")
                self._create_directory(f"{root}/hyperparameters/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}")
                self._create_directory(f"{root}/hyperparameters/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds")
                self._create_directory(f"{root}/hyperparameters/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}")
                # Serializing json
                json_object = json.dumps(hyperparameters, indent=4, ensure_ascii=False)

                with open(f"{root}/hyperparameters/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}/fold{k}.json", "w", encoding='utf-8') as outfile:
                    outfile.write(json_object)
                
                print("hyperparameters")
                
                print("test_metrics")

                # Serializing json
                json_object = json.dumps(test_metrics, indent=4, ensure_ascii=False)

                # Writing to sample.json
                self._create_directory(f"{root}/metrics")
                self._create_directory(f"{root}/metrics/{self.validation_name}")
                self._create_directory(f"{root}/metrics/{self.validation_name}/{self.input}")
                self._create_directory(f"{root}/metrics/{self.validation_name}/{self.input}/{self.max_length}")
                self._create_directory(f"{root}/metrics/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}")
                self._create_directory(f"{root}/metrics/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds")
                self._create_directory(f"{root}/metrics/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}")

                with open(f"{root}/metrics/{self.validation_name}/{self.input}/{self.max_length}/{self.corpus}/{self.folds}folds/{bert_name}/fold{k}.json", "w", encoding='utf-8') as outfile:
                    outfile.write(json_object)

if __name__ == '__main__':
    #Hiperparameters

    evaluation_strategy = "epoch" 
    eval_steps = None 
    save_total_limit = 5 
    save_strategy = "epoch"

    learning_rate = 2e-5 
    weight_decay = 0 

    batch_size = 8
    num_train_epochs = 4 
    metric_for_best_model = "f1_macro" 

    adam_beta1 = 0.9
    adam_beta2 = 0.999


    bert_list = {
        "bertikal" : "felipemaiapolo/legalnlp-bert", 
        "bertimbau": "neuralmind/bert-base-portuguese-cased",
        }
    
    corpus = "without_exatas_sociais" # or "standard"

    bert_list = {
        "bertikal" : "felipemaiapolo/legalnlp-bert", 
        }

    cross_validation = Cross_validation(corpus=corpus, bert_list=bert_list, validation_name="cross_validation", input="ementa", max_length=75)
    cross_validation.run(evaluation_strategy, eval_steps, save_total_limit, save_strategy,learning_rate,weight_decay, batch_size, num_train_epochs, metric_for_best_model, adam_beta1, adam_beta2)
