import pandas as pd
import numpy as np
from scipy.signal import decimate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import os
import csv

def get_merged_data(meta_subset):
    merged = [row["data"] for _, row in meta_subset.iterrows()]
    return pd.concat(merged, ignore_index=True) if merged else pd.DataFrame()

def clean(df):
    df = df.dropna(subset=["class"])
    return df[~df["class"].isin(["Error", "Synchronization", "None"])]

def create_sequences(df, seq_len, data_columns):
    data, labels = [], []
    arr = df[data_columns].values
    target = df["class"].values
    for i in range(len(df) - seq_len):
        seq = arr[i:i + seq_len]
        label = target[i + seq_len - 1]
        data.append(seq)
        labels.append(label)
    return np.array(data), np.array(labels)

def downsample_signal(data_series, factor, filter_order):
    return decimate(data_series, q=factor, n=filter_order, ftype='iir')

def log_to_csv(data_dict, csv_path):
    fieldnames = list(data_dict.keys())
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)

def plot_results(train_acc, val_acc, train_loss, val_loss, val_f1, val_mcc, training_id):
    plt.figure(figsize=(24, 5))
    plt.subplot(1, 4, 1)
    plt.plot(train_acc, label='Training Acc')
    plt.plot(val_acc, label='Validation Acc')
    plt.title('Accuracy vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 2)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 3)
    plt.plot(val_f1, label='Validation F1 Score', color='green')
    plt.title('Validation F1 Score vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 4, 4)
    plt.plot(val_mcc, label='Validation MCC', color='purple')
    plt.title('Validation MCC vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MCC')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    os.makedirs("Plots/SPARL3", exist_ok=True)
    plt.savefig(f"Plots/SPARL3/training_hist_{training_id}.png")
    plt.close()


def print_confusion_matrix(y_true, y_pred, le, training_id):
    present_indices = sorted(list(set(y_true) | set(y_pred)))
    if not present_indices: return
    present_names = le.inverse_transform(present_indices)
    cm = confusion_matrix(y_true, y_pred, labels=present_indices)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=present_names, yticklabels=present_names,
                cbar_kws={'label': 'Prediction %'})
    plt.title(f"Normalized Confusion Matrix - {training_id}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs("Plots/SPARL3", exist_ok=True)
    plt.savefig(f"Plots/SPARL3/confusion_matrix_{training_id}.png")
    plt.close()

def map_to_superclass(encoded_labels, encoder, mapping):
    decoded_labels = encoder.inverse_transform(encoded_labels)
    return np.array([mapping.get(label, "Unmapped") for label in decoded_labels])

def print_superclass_cm(y_true, y_pred, le, mapping, training_id):
    y_true_super = map_to_superclass(y_true, le, mapping)
    y_pred_super = map_to_superclass(y_pred, le, mapping)
    
    present_labels = sorted(list(set(y_true_super) | set(y_pred_super)))
    if not present_labels: return
    cm = confusion_matrix(y_true_super, y_pred_super, labels=present_labels)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-9) * 100
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt=".1f", cmap="Blues",
                xticklabels=present_labels, yticklabels=present_labels,
                cbar_kws={'label': 'Prediction %'})
    plt.title(f"Superclass Confusion Matrix - {training_id}")
    plt.ylabel("True Superclass")
    plt.xlabel("Predicted Superclass")
    plt.tight_layout()
    os.makedirs("Plots/SPARL3", exist_ok=True)
    plt.savefig(f"Plots/SPARL3/sc_confusion_matrix_{training_id}.png")
    plt.close()



def plot_class_distribution_pie_chart(df, title="Class distribution"):
    # concat all 'class' columns from series of dataframes
    concat_df = pd.concat([row['data']['class'] for _, row in df.iterrows()], ignore_index=True)
    # remove None, Synchronization, Error
    concat_df = concat_df[~concat_df.isin(['None', 'Synchronization', 'Error'])]
    class_counts = concat_df.value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    #plt.title(title)
    plt.axis('equal')
    plt.show()

def plot_class_distribution_bar_chart(df, title="Class distribution"):
    concat_df = pd.concat([row['data']['class'] for _, row in df.iterrows()], ignore_index=True)
    concat_df = concat_df[~concat_df.isin(['None', 'Synchronization', 'Error'])]
    class_counts = concat_df.value_counts()
    colors = plt.cm.tab20.colors
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(class_counts)), class_counts.values, color=colors[:len(class_counts)])
    plt.xlabel('Class', fontsize=18)
    plt.ylabel('Label Count', fontsize=18)
    plt.xticks([])  # X-Ticks ausblenden
    plt.yticks(fontsize=18)
    plt.grid(axis='y')
    plt.tight_layout()
    # Legende hinzufügen
    labels = class_counts.index.astype(str)
    for bar, label in zip(bars, labels):
        bar.set_label(label)
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height, f'{int(height)}',
                 ha='center', va='bottom', fontsize=18)
    plt.legend(title="Classes", fontsize=18, title_fontsize=20)
    plt.show()



def plot_columns_time_series(df, column_names, title=None):
    n = len(column_names)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, col in enumerate(column_names):
        axes[i].plot(df[col].values)
        axes[i].set_ylabel(col)
        axes[i].grid(True)
    axes[-1].set_xlabel("Zeitindex")
    if title:
        plt.suptitle(title)
    else:
        plt.suptitle("Zeitreihen für Spalten: " + ", ".join(column_names))
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def jitter(x, sigma=0.05):
    noise = np.random.normal(loc=0., scale=sigma, size=x.shape)
    return x + noise

def scaling(x, sigma=0.1):
    scaling_factor = np.random.normal(loc=1.0, scale=sigma, size=(1, x.shape[1]))
    return x * scaling_factor


