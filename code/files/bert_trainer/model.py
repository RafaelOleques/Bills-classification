from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from transformers import EvalPrediction
from transformers import EarlyStoppingCallback, IntervalStrategy

import numpy as np

from sklearn.metrics import f1_score, fbeta_score, roc_auc_score, accuracy_score, precision_score, recall_score, average_precision_score
from sklearn.metrics import coverage_error
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import label_ranking_loss

import torch
from datasets import Dataset, DatasetDict

class Model:
    def __init__(self, bert_version, labels, id2label, label2id, early_stopping_patience=3, max_length=512, padding="max_length", truncation=True, input="ementa", output="temas") -> None:
        #Hiperparameters
        self.labels = labels
        self.id2label = id2label
        self.label2id = label2id

        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        self.input = input
        self.output = output

        self.early_stopping_patience = early_stopping_patience
        
        #Model
        self.tokenizer = AutoTokenizer.from_pretrained(bert_version)
        self.model = self.set_model(bert_version=bert_version)

        #To set after
        self.args = None
        self.encoded_dataset = None
        self.trainer = None

    def preprocess_data(self, examples):
        # take a batch of texts
        text = examples[self.input]
        # encode them
        encoding = self.tokenizer(text, padding=self.padding, truncation=self.truncation, max_length=self.max_length)
        # add labels
        labels_batch = {label: [] for label in self.labels}

        for label in self.labels:
            labels_batch[label] = [label in examples_labels for examples_labels in examples[self.output]]
        # create numpy array of shape (batch_size, num_labels)
        labels_matrix = np.zeros((len(text), len(self.labels)))
        # fill numpy array
        for idx, label in enumerate(self.labels):
            labels_matrix[:, idx] = labels_batch[label]
        
        encoding["labels"] = labels_matrix.tolist()

        return encoding

    def set_dataset(self, train_dataset, validation_dataset, test_dataset):
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset})

        self.encoded_dataset = dataset.map(self.preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

        self.encoded_dataset.set_format("torch")

    def set_dataset_manual(self, train_dataset, validation_dataset, test_dataset):
        dataset = DatasetDict({
            'train': train_dataset,
            'validation': validation_dataset,
            'test': test_dataset})

        self.encoded_dataset = dataset.map(self.preprocess_data, batched=True, remove_columns=dataset['train'].column_names)

        self.encoded_dataset.set_format("torch")

    def set_model(self, bert_version, problem_type="multi_label_classification"):
        model = AutoModelForSequenceClassification.from_pretrained(bert_version, 
                                                           problem_type=problem_type, 
                                                           num_labels=len(self.labels),
                                                           id2label=self.id2label,
                                                           label2id=self.label2id)

        return model

    def training_arguments(self, name, evaluation_strategy, eval_steps, save_total_limit, save_strategy, 
                                learning_rate, weight_decay, 
                                batch_size, num_train_epochs, metric_for_best_model, 
                                adam_beta1, adam_beta2):
        self.args = TrainingArguments(
            name,
            evaluation_strategy = evaluation_strategy,
            eval_steps=eval_steps,
            save_total_limit =save_total_limit,
            save_strategy = save_strategy,
            learning_rate = learning_rate,
            per_device_train_batch_size = batch_size,
            per_device_eval_batch_size = batch_size,
            num_train_epochs = num_train_epochs,
            weight_decay = weight_decay,
            load_best_model_at_end = True,
            metric_for_best_model=metric_for_best_model,
            adam_beta1 = adam_beta1,
            adam_beta2 = adam_beta2,
        )

    def multi_label_metrics(self, predictions, labels, threshold=0.5):
        # first, apply sigmoid on predictions which are of shape (batch_size, num_labels)
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.Tensor(predictions))
        
        # next, use threshold to turn them into integer predictions
        y_pred = np.zeros(probs.shape)
        y_pred[np.where(probs >= threshold)] = 1
        
        # finally, compute metrics
        y_true = labels
        
        #1
        f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
        f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
        f1_weighted_average = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
                
        #2
        precision = precision_score(y_true, y_pred, average = 'macro')
        recall = recall_score(y_true, y_pred, average = 'macro')
        
        #3
        accuracy = accuracy_score(y_true, y_pred)
                
        #Per class    
        f1_per_class = f1_score(y_true=y_true, y_pred=y_pred, average=None)
        precision_per_class = precision_score(y_true, y_pred, average = None)
        recall_per_class = recall_score(y_true, y_pred, average = None)
        
        # return as dictionary
        metrics = {
                    'f1_macro': f1_macro_average,
                    'f1_micro': f1_micro_average,
                    'f1_weighted': f1_weighted_average,                               
                    'recall': recall,
                    'precision': precision,
                    'accuracy': accuracy
                    }
        
        for idx, val in enumerate(f1_per_class):
            metrics[f"{self.id2label[idx]}_f1"] = f1_per_class[idx]
            metrics[f"{self.id2label[idx]}_precision"] = precision_per_class[idx]
            metrics[f"{self.id2label[idx]}_recall"] = recall_per_class[idx]
        
        return metrics

    def compute_metrics(self, p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, 
                tuple) else p.predictions
        
        result = self.multi_label_metrics(
            predictions=preds, 
            labels=p.label_ids)
        
        return result

    def set_trainer(self):
        self.trainer = Trainer(
                self.model,
                self.args,
                train_dataset=self.encoded_dataset["train"],
                eval_dataset=self.encoded_dataset["validation"],
                tokenizer=self.tokenizer,
                compute_metrics=self.compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=self.early_stopping_patience)]
            )
        

    def train(self):
        self.set_trainer()
        self.trainer.train()

    def get_metrics_validation(self):
        metrics = self.trainer.evaluate()

        return metrics

    def get_probs_and_metrics_test(self):
        y_probs, labels_ids, metrics = self.trainer.predict(self.encoded_dataset["test"])

        return y_probs, metrics
    
    def save_model(self, path):
        self.trainer.save_model(path)

'''
init
set_dataset
training_arguments
train
'''