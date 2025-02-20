# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================
import os
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from transformers import RobertaModel, RobertaTokenizer

logging.basicConfig(level=logging.ERROR)



class SentimentData(Dataset):
    """
    Custom Dataset class to tokenize and encode input text.
    """
    def __init__(self, dataframe, tokenizer, max_len, padding, TEXT_COLUMN, TARGET_COLUMN):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe[TEXT_COLUMN]
        self.targets = dataframe[TARGET_COLUMN]
        self.max_len = max_len
        self.padding = padding

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text.iloc[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding=self.padding,
            truncation=True,
            return_tensors="pt"
        )

        # Remove the batch dimension.
        ids = inputs['input_ids'].squeeze(0)
        mask = inputs['attention_mask'].squeeze(0)

        return {
            'ids': ids,
            'mask': mask,
            'targets': torch.tensor(self.targets.iloc[index], dtype=torch.float)
        }

# =============================================================================
# 5. ROBERTA TRAINER CLASS DEFINITION
# =============================================================================
class RobertaTrainer:
    def __init__(self, df, tokenizer, max_len, train_batch_size, valid_batch_size,
                 test_size, device, padding, text_column, target_column, random_state):
        self.df = df.copy()
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.test_size = test_size
        self.device = device
        self.padding = padding
        self.text_column = text_column
        self.target_column = target_column
        self.random_state = random_state

        self.training_loader = None
        self.testing_loader = None
        self.model = None

    def tokenize_data(self):
        """
        Tokenize and split the data into training and testing sets.
        The split is done as follows:
          1. Separate synthetic and nonsynthetic data.
          2. Perform a stratified split on nonsynthetic data using the clean_text column.
          3. Add synthetic data to the training set.
        """
        # Separate synthetic and nonsynthetic data.
        self.df_synthetic = self.df[self.df['synthetic'] == 1]
        self.df_nonsynthetic = self.df[self.df['synthetic'] == 0]

        # Perform stratified split on nonsynthetic data.
        X_train, X_val, y_train, y_val = train_test_split(
            self.df_nonsynthetic[self.text_column],
            self.df_nonsynthetic[self.target_column],
            test_size=self.test_size,
            shuffle=True,
            random_state=self.random_state,
            stratify=self.df_nonsynthetic[self.target_column]
        )

        # Add synthetic data to the training set.
        X_train = pd.concat([X_train, self.df_synthetic[self.text_column]])
        y_train = pd.concat([y_train, self.df_synthetic[self.target_column]])

        print("TRAIN Dataset shape:", X_train.shape)
        print("TEST Dataset shape:", X_val.shape)

        # Create DataFrames for train and test.
        train_df = pd.DataFrame({self.text_column: X_train, self.target_column: y_train})
        test_df = pd.DataFrame({self.text_column: X_val, self.target_column: y_val})

        # Create Dataset objects.
        training_set = SentimentData(train_df, self.tokenizer, self.max_len, self.padding, self.text_column, self.target_column)
        testing_set = SentimentData(test_df, self.tokenizer, self.max_len, self.padding, self.text_column, self.target_column)

        # Create DataLoaders.
        self.training_loader = DataLoader(
            training_set,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=0
        )
        self.testing_loader = DataLoader(
            testing_set,
            batch_size=self.valid_batch_size,
            shuffle=True,
            num_workers=0
        )
        return self.training_loader, self.testing_loader

    def build_model(self):
        """
        Build and return the RoBERTa-based classification model.
        """
        class RobertaClass(torch.nn.Module):
            def __init__(self):
                super(RobertaClass, self).__init__()
                self.l1 = RobertaModel.from_pretrained("distilroberta-base")
                self.pre_classifier = torch.nn.Linear(768, 768)
                self.dropout = torch.nn.Dropout(0.3)
                self.classifier = torch.nn.Linear(768, 1)

            def forward(self, input_ids, attention_mask):
                output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
                hidden_state = output_1[0]
                pooler = hidden_state[:, 0]  # Use the [CLS] token representation.
                pooler = self.pre_classifier(pooler)
                pooler = torch.nn.ReLU()(pooler)
                pooler = self.dropout(pooler)
                output = self.classifier(pooler)
                return output

        self.model = RobertaClass()
        self.model.to(self.device)
        return self.model

    def train_model(self, epochs, learning_rate):
        """
        Train the model for a specified number of epochs.
        """
        loss_function = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=learning_rate)

        def calculate_accuracy(preds, targets):
            return (preds == targets).sum().item()

        for epoch in range(epochs):
            tr_loss = 0
            n_correct = 0
            nb_tr_steps = 0
            nb_tr_examples = 0
            self.model.train()

            for step, data in enumerate(tqdm(self.training_loader, desc="Training")):
                ids = data['ids'].to(self.device, dtype=torch.long)
                mask = data['mask'].to(self.device, dtype=torch.long)
                targets = data['targets'].to(self.device, dtype=torch.float)
                outputs = self.model(ids, mask).squeeze(1)
                loss = loss_function(outputs, targets)
                tr_loss += loss.item()

                # Convert logits to probabilities.
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).long()
                n_correct += calculate_accuracy(preds, targets.long())
                nb_tr_steps += 1
                nb_tr_examples += targets.size(0)

                if step % 50 == 0:
                    loss_step = tr_loss / nb_tr_steps
                    accu_step = (n_correct * 100) / nb_tr_examples
                    print(f"Step {step}: Training Loss: {loss_step:.4f}, Training Accuracy: {accu_step:.2f}%")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = tr_loss / nb_tr_steps
            epoch_accu = (n_correct * 100) / nb_tr_examples
            print(f"Epoch {epoch + 1}: Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_accu:.2f}%")
        return self.model

    def evaluate_model(self):
        """
        Evaluate the model on the testing set and print accuracy, precision, and recall.
        """
        self.model.eval()
        n_correct = 0
        nb_steps = 0
        nb_examples = 0
        all_targets = []
        all_predictions = []
        loss_function = torch.nn.BCEWithLogitsLoss()
        tr_loss = 0

        for step, data in enumerate(tqdm(self.testing_loader, desc="Validation")):
            ids = data['ids'].to(self.device, dtype=torch.long)
            mask = data['mask'].to(self.device, dtype=torch.long)
            targets = data['targets'].to(self.device, dtype=torch.float)
            outputs = self.model(ids, mask).squeeze(1)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).long()

            n_correct += (preds == targets.long()).sum().item()
            nb_steps += 1
            nb_examples += targets.size(0)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(preds.cpu().numpy())

        avg_val_loss = tr_loss / nb_steps
        accuracy = (n_correct * 100) / nb_examples
        precision = precision_score(all_targets, all_predictions, average="binary")
        recall = recall_score(all_targets, all_predictions, average="binary")

        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")
        print(f"Precision: {precision:.2f}, Recall: {recall:.2f}")
        return accuracy, precision, recall


    def save_model(self, output_model_file, output_tokenizer_dir):
        """
        Save the model's state dictionary and the tokenizer's vocabulary.
        
        Parameters:
            output_model_file (str): Path to the file where the model state will be saved.
            output_tokenizer_dir (str): Directory where the tokenizer will be saved.
            
        Note: Since the model class is defined locally inside build_model,
              we save only the state_dict. To load the model later, you'll need to 
              instantiate the model (e.g., via build_model) and load this state_dict.
        """
        # Save the model state dictionary
        torch.save(self.model.state_dict(), output_model_file)
        print(f"Model state_dict saved to {output_model_file}")

        # Save the tokenizer
        # Depending on your tokenizer version, you might use `save_pretrained` instead.
        self.tokenizer.save_vocabulary(output_tokenizer_dir)
        print(f"Tokenizer vocabulary saved to {output_tokenizer_dir}")