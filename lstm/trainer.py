import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, matthews_corrcoef, f1_score, accuracy_score
import os
from utils import plot_results, print_confusion_matrix, print_superclass_cm, log_to_csv
from torch.optim.lr_scheduler import ReduceLROnPlateau

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, le, optimizer, criterion, config, trial=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.le = le
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': [], 'lr_history': [], 'mcc_epoch': []}
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.best_vall_loss = float('inf')
        self.best_val_f1 = 0.0
        self.scheduler = None
        self.trial = trial

        if self.config.SCHEDULER_ENABLED:
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, 
                mode='min',
                factor=self.config.SCHEDULER_FACTOR,
                patience=self.config.SCHEDULER_PATIENCE
            )

    def _plot_samples(self, batch):
        sample = batch[0].detach().cpu().numpy()  # (7, seq_len)

        fig, axes = plt.subplots(7, 1, figsize=(10, 12), sharex=True)
        for i in range(7):
            axes[i].plot(sample.T[i])
            axes[i].set_ylabel(f"Sensor {i + 1}")
            axes[i].set_ylim(-1, 1)
        axes[-1].set_xlabel("Zeit")
        plt.tight_layout()
        plt.show()
    def _train_epoch(self):
        self.model.train()
        running_loss, correct, total = 0.0, 0, 0
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for X_batch, y_batch in pbar:
            X_batch, y_batch = X_batch.to(self.config.DEVICE), y_batch.to(self.config.DEVICE)
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        return running_loss / len(self.train_loader.dataset), correct / total

    def _validate_epoch(self):
        self.model.eval()
        running_loss, correct, total = 0.0, 0, 0
        all_preds, all_true = [], []
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for X_batch, y_batch in pbar:
                #self._plot_samples(X_batch)
                X_batch, y_batch = X_batch.to(self.config.DEVICE), y_batch.to(self.config.DEVICE)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                total += y_batch.size(0)
                correct += (predicted == y_batch).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())
            val_loss = running_loss / len(self.val_loader.dataset)
            val_acc = correct/total
            # chose macro for average to treat each class equally without weighting 
            # and to prevent previous behavior of overclassifying majority classes
            val_f1 = f1_score(all_true, all_preds, average='macro', zero_division=0)
            mcc_epoch = matthews_corrcoef(all_true, all_preds)

        # return running_loss / len(self.val_loader.dataset), correct / total
        return val_loss, val_acc, val_f1, mcc_epoch

    def train(self, training_id):
        print(f"Starting training run {training_id} for {self.config.EPOCHS} epochs on {self.config.DEVICE}...")
        outer_pbar = tqdm(range(self.config.EPOCHS), desc="Total Progress")
        pruning_suggested = False
        for epoch in outer_pbar:
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc, val_f1, mcc_epoch = self._validate_epoch()

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1)
            self.history['mcc_epoch'].append(mcc_epoch)

            outer_pbar.set_description(f"Epoch {epoch+1} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1 (Macro): {val_f1:.4f} | Val Mcc: {mcc_epoch:.4f}")
            if self.trial:
                self.trial.report(mcc_epoch, epoch)
                if self.trial.should_prune():
                    print(f"Optuna suggests pruning at epoch {epoch+1} due to unpromising result")
                    pruning_suggested = True
                    break
            if self.scheduler:
                self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            self.history['lr_history'].append(current_lr)
            # if val_f1 > self.best_val_f1:
            #     self.best_val_f1 = val_f1
            #     self._save_checkpoint(training_id)
            if self.config.EARLY_STOPPING_ENABLED:
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.patience_counter = 0
                    self._save_checkpoint(training_id)
                else:
                    self.patience_counter+=1
                if self.patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                    print(f"Validation loss has not improved for {self.config.EARLY_STOPPING_PATIENCE} epochs. Stopping early")
                    break
        print("\nTraining finished.")
        return pruning_suggested

    def _save_checkpoint(self, training_id):
        model_path = f"Models/SPARL3/{self.config.MODEL_NAME}_{training_id}_best.pth"
        torch.save(self.model.state_dict(), model_path)

    def _get_checkpoint_path(self, training_id):
        return f"Models/SPARL3/{self.config.MODEL_NAME}_{training_id}_best.pth"

    def evaluate(self, training_id, csv_path):
        print("\nEvaluating on the test set...")
        self.model.eval()
        all_preds, all_true = [], []
        best_model_path = self._get_checkpoint_path(training_id)
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.config.DEVICE))
        else:
            print("No checkpoint found")
        with torch.no_grad():
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.config.DEVICE), y_batch.to(self.config.DEVICE)
                outputs = self.model(X_batch)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())
        
        print("\n--- Final Test Classification Report ---")
        print(classification_report(all_true, all_preds, target_names=self.le.classes_))

        os.makedirs("Pred/SPARL3", exist_ok=True)
        np.save(f"Pred/SPARL3/predictions_{training_id}.npy", all_preds)

        plot_results(self.history['train_acc'], self.history['val_acc'], self.history['train_loss'], self.history['val_loss'], self.history['val_f1'], self.history["mcc_epoch"] ,training_id)

        if self.config.USE_SUPERCLASSES:
            print_confusion_matrix(all_true, all_preds, self.le, training_id)
        else:
            print_confusion_matrix(all_true, all_preds, self.le, training_id)
            print_superclass_cm(all_true, all_preds, self.le, self.config.SUPERCLASS_MAPPING, training_id)

        mcc = matthews_corrcoef(all_true, all_preds)
        acc = accuracy_score(all_true, all_preds)
        f1_macro = f1_score(all_true, all_preds, average='macro')
        summary = {
            'training_id': training_id,
            'learning_rate': self.config.LEARNING_RATE,
            'num_cnn_layers': self.config.NUM_CNN_LAYERS,
            'best_val_acc': self.best_val_acc,
            'min_val_loss': min(self.history['val_loss']) if self.history['val_loss'] else float('inf'),
            'best_val_f1': self.best_val_f1,
            'final_train_acc': self.history['train_acc'][-1] if self.history['train_acc'] else 0,
            'min_train_loss': min(self.history['train_loss']) if self.history['train_loss'] else float('inf'),
            'batch_size': self.config.BATCH_SIZE, 'targeted_freq': self.config.TARGET_FREQ,
            'dataset': "SPARL3",
            'mcc' : mcc,
            "test_acc": acc,
            "test_f1_macro": f1_macro
        }
        log_to_csv(summary, csv_path)
        return summary