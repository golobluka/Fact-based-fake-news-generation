import os
import time
import datetime
import random
import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score


class BertTrainer:
    def __init__(
        self,
        df,
        text_column="text",
        label_column="target",
        model_name="bert-base-uncased",
        max_length=128,
        batch_size=32,
        random_state=42,
        device=None
    ):
        """
        Initialize the trainer with a pandas DataFrame and basic parameters.

        Parameters:
          - df: DataFrame containing your data.
          - text_column: column name with the input text.
          - label_column: column name with the target labels.
          - model_name: name of the pretrained BERT model.
          - max_length: maximum sequence length for tokenization.
          - batch_size: batch size for DataLoaders.
          - random_state: random seed.
          - device: PyTorch device (if None, it automatically selects GPU if available).
        """
        self.df = df.copy()
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.batch_size = batch_size
        self.random_state = random_state
        self.labels = df[self.label_column].values

        # Set device automatically if not provided
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # Load the tokenizer
        print("Loading BERT tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=True)
        self.model_name = model_name

        # Placeholders for tokenized outputs and DataLoaders
        self.input_ids = None
        self.attention_masks = None
        self.train_dataloader = None
        self.validation_dataloader = None

        # Placeholder for the model
        self.model = None

    def tokenize_dataset(
        self,
        add_special_tokens=True,
        padding="max_length",
        truncation=True,
    ):
        """
        Tokenizes all sentences in the dataset.

        Parameters:
          - add_special_tokens: whether to add [CLS] and [SEP] tokens.
          - padding: padding strategy (e.g., "max_length").
          - truncation: whether to truncate sentences longer than max_length.
        """
        sentences = self.df[self.text_column].values
        input_ids = []
        attention_masks = []

        print("Tokenizing {} sentences...".format(len(sentences)))
        for sent in sentences:
            encoded_dict = self.tokenizer.encode_plus(
                sent,
                add_special_tokens=add_special_tokens,
                max_length=self.max_length,
                padding=padding,
                truncation=truncation,
                # You can add more arguments if needed (e.g., return_token_type_ids)
            )
            input_ids.append(encoded_dict["input_ids"])
            attention_masks.append(encoded_dict["attention_mask"])

        self.input_ids = input_ids
        self.attention_masks = attention_masks

        print("Tokenization complete.")
        return input_ids, attention_masks

    def prepare_dataloader(self, test_size=0.1):
        """
        Splits tokenized data into training and validation sets with stratified sampling.
        If a 'synthetic' column exists in the DataFrame, synthetic examples are
        added to the training set after splitting the nonsynthetic data.
    
        Parameters:
          - test_size: proportion of nonsynthetic data to use for validation.
        """
        from sklearn.model_selection import train_test_split
        from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
    
        if "synthetic" in self.df.columns:
            # Separate synthetic and nonsynthetic data.
            self.df_synthetic = self.df[self.df["synthetic"] == 1]
            self.df_nonsynthetic = self.df[self.df["synthetic"] == 0]
    
            # Get indices for nonsynthetic and synthetic data.
            nonsyn_idx = self.df_nonsynthetic.index.tolist()
            synthetic_idx = self.df_synthetic.index.tolist()
    
            # Filter tokenized outputs and labels for nonsynthetic data.
            nonsyn_input_ids = [self.input_ids[i] for i in nonsyn_idx]
            nonsyn_attention_masks = [self.attention_masks[i] for i in nonsyn_idx]
            nonsyn_labels = self.df_nonsynthetic[self.label_column].values
    
            # Stratified split on nonsynthetic data.
            X_train_nonsyn, X_val, y_train_nonsyn, y_val = train_test_split(
                nonsyn_input_ids,
                nonsyn_labels,
                test_size=test_size,
                shuffle=True,
                random_state=self.random_state,
                stratify=nonsyn_labels
            )
            train_masks_nonsyn, val_masks, _, _ = train_test_split(
                nonsyn_attention_masks,
                nonsyn_labels,
                test_size=test_size,
                shuffle=True,
                random_state=self.random_state,
                stratify=nonsyn_labels
            )
    
            # Prepare synthetic data (all synthetic examples go to training).
            synthetic_input_ids = [self.input_ids[i] for i in synthetic_idx]
            synthetic_attention_masks = [self.attention_masks[i] for i in synthetic_idx]
            synthetic_labels = self.df_synthetic[self.label_column].values
    
            # Combine nonsynthetic training data with all synthetic data.
            X_train = X_train_nonsyn + synthetic_input_ids
            train_masks = train_masks_nonsyn + synthetic_attention_masks
            y_train = np.concatenate([y_train_nonsyn, synthetic_labels])
        else:
            # If no synthetic column exists, use the entire dataset.
            X_train, X_val, y_train, y_val = train_test_split(
                self.input_ids,
                self.labels,
                test_size=test_size,
                shuffle=True,
                random_state=self.random_state,
                stratify=self.labels  # Stratified split on the full dataset.
            )
            train_masks, val_masks, _, _ = train_test_split(
                self.attention_masks,
                self.labels,
                test_size=test_size,
                shuffle=True,
                random_state=self.random_state,
                stratify=self.labels
            )
    
        # Convert lists to torch tensors.
        train_inputs = torch.tensor(X_train)
        val_inputs = torch.tensor(X_val)
        train_labels = torch.tensor(y_train)
        val_labels = torch.tensor(y_val)
        train_masks = torch.tensor(train_masks)
        val_masks = torch.tensor(val_masks)
    
        # Create TensorDatasets.
        from torch.utils.data import TensorDataset
        train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
        val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    
        # Create DataLoaders.
        train_sampler = RandomSampler(train_dataset)
        val_sampler = SequentialSampler(val_dataset)
        self.train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.batch_size)
        self.validation_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=self.batch_size)
    
        print("DataLoaders created:")
        print("  Training batches:", len(self.train_dataloader))
        print("  Validation batches:", len(self.validation_dataloader))
        return self.train_dataloader, self.validation_dataloader
    
    def build_model(
        self,
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    ):
        """
        Loads the BERT model for sequence classification.

        Parameters:
          - num_labels: number of output labels.
          - output_attentions: whether to return attention weights.
          - output_hidden_states: whether to return hidden states.
        """
        print("Loading BERT model for sequence classification...")
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        self.model.to(self.device)
        print("Model loaded on device:", self.device)
        return self.model

    def train_model(
        self,
        epochs=4,
        learning_rate=2e-5,
        epsilon=1e-8,
    ):
        """
        Trains the BERT model using the training DataLoader and evaluates on the validation DataLoader.

        Parameters:
          - epochs: number of training epochs.
          - learning_rate: learning rate for the optimizer.
          - epsilon: epsilon value for the AdamW optimizer.
        """
        # Create optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=epsilon)
        total_steps = len(self.train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

        # Set seed for reproducibility
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.random_state)

        # Helper functions
        def format_time(elapsed):
            return str(datetime.timedelta(seconds=int(round(elapsed))))

        def flat_accuracy(preds, labels):
            pred_flat = np.argmax(preds, axis=1).flatten()
            labels_flat = labels.flatten()
            return np.sum(pred_flat == labels_flat) / len(labels_flat)

        loss_values = []

        # Training loop
        for epoch_i in range(epochs):
            print("")
            print("======== Epoch {} / {} ========".format(epoch_i + 1, epochs))
            print("Training...")

            t0 = time.time()
            total_loss = 0
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                if step % 40 == 0 and step != 0:
                    elapsed = format_time(time.time() - t0)
                    print("  Batch {:>5,}  of  {:>5,}. Elapsed: {}.".format(step, len(self.train_dataloader), elapsed))

                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs[0]
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss / len(self.train_dataloader)
            loss_values.append(avg_train_loss)
            print("  Average training loss: {:.2f}".format(avg_train_loss))
            print("  Training epoch took: {}".format(format_time(time.time() - t0)))

            # Validation after each epoch
            print("Running Validation...")
            t0 = time.time()
            self.model.eval()
            eval_accuracy = 0
            nb_eval_steps = 0

            for batch in self.validation_dataloader:
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
                logits = outputs[0].detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()
                eval_accuracy += flat_accuracy(logits, label_ids)
                nb_eval_steps += 1

            print("  Validation Accuracy: {:.2f}".format(eval_accuracy / nb_eval_steps))
            print("  Validation took: {}".format(format_time(time.time() - t0)))

        print("")
        print("Training complete!")
        return loss_values

    def evaluate_model(self):
        """
        Evaluates the trained model on the validation set, reporting accuracy, precision, and recall.
        """
        self.model.eval()
        true_labels = []
        pred_labels = []

        for batch in self.validation_dataloader:
            batch = tuple(t.to(self.device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = self.model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            true_labels.extend(label_ids)
            pred_labels.extend(np.argmax(logits, axis=1))

        accuracy = accuracy_score(true_labels, pred_labels)
        precision = precision_score(true_labels, pred_labels, average="binary")
        recall = recall_score(true_labels, pred_labels, average="binary")

        print("Evaluation Results:")
        print("  Accuracy: {:.2f}".format(accuracy))
        print("  Precision: {:.2f}".format(precision))
        print("  Recall: {:.2f}".format(recall))

        return accuracy, precision, recall
