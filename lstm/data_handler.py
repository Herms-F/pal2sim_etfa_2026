import pickle

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader

from utils import get_merged_data, clean, downsample_signal, jitter, scaling

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


class SensorDataset(Dataset):
    def __init__(self, df, seq_len, sensor_cols, augment=False, aug_prob=0.5):
        self.df = df.reset_index(drop=True)
        self.seq_len = seq_len
        self.sensor_cols = sensor_cols
        self.data = self.df[self.sensor_cols].values
        self.labels = self.df['class'].values
        self.augment = augment
        self.aug_prob = aug_prob

    def __len__(self):
        return len(self.df) - self.seq_len

    def __getitem__(self, idx):
        x_np = self.data[idx : idx + self.seq_len]
        y = self.labels[idx + self.seq_len - 1]

        if self.augment and np.random.rand() < self.aug_prob:
            if np.random.rand() < 0.5:
                x_np = jitter(x_np)
            else: 
                x_np = scaling(x_np)
        return torch.tensor(x_np, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def plot_label_distribution(self, dataset_name="Dataset"):
        labels, counts = np.unique(self.labels, return_counts=True)
        plt.figure(figsize=(10, 6))
        plt.bar(labels, counts, color='skyblue')
        plt.xlabel('Klassenlabel')
        plt.ylabel('Anzahl der Sequenzen')
        plt.title('Verteilung der Klassenlabels im Datensatz ' + dataset_name)
        plt.xticks(labels)
        plt.grid(axis='y')
        plt.show()

    def plot_sequence(self, idx):
        x, y = self.__getitem__(idx)
        seq_len = x.shape[0]
        fig, axes = plt.subplots(len(self.sensor_cols), 1, figsize=(10, 2 * len(self.sensor_cols)), sharex=True)
        if len(self.sensor_cols) == 1:
            axes = [axes]
        for i, col in enumerate(self.sensor_cols):
            axes[i].plot(range(seq_len), x[:, i].numpy())
            axes[i].set_ylabel(col)
            axes[i].set_ylim(-1, 1)
            axes[i].grid(True)
        axes[-1].set_xlabel("Zeit")
        plt.suptitle(f"Label: {y.item()}")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.show()

def deritivate_of_signal(signal):
    deriv = np.diff(signal, prepend=signal[0])
    return deriv

def calc_norm_entropy(labels):
    value, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))
    norm_entropy = entropy / np.log2(len(value) + 1e-9)
    return norm_entropy


class DataHandler:
    def __init__(self, config):
        self.config = config
        self.le = None
        self.scaler = None

    def get_data_loaders(self):
        print("Starting data preparation...", flush=True)
        with open(self.config.DATASET_PATH, "rb") as f:
            metadata = pickle.load(f)
        
        df_final = metadata.copy()

        if not self.config.IS_WITHOUT_DS:
            print("Downsampling active!")
            results = []
            for _, row in metadata.iterrows():
                df_raw = row['data']
                downsampled_data = {}
                for col in self.config.SENSOR_COLS:
                    if col in df_raw.columns:
                        downsampled_data[col] = downsample_signal(
                            df_raw[col].values, self.config.DS_FACTOR, self.config.FILTER_ORDER
                        )
                if 'class' in df_raw.columns:
                     downsampled_data['class'] = df_raw['class'].values[::self.config.DS_FACTOR]
                results.append(pd.DataFrame(downsampled_data))
            df_final['data'] = results

        test_mask = df_final['experiment'] == self.config.TEST_EXPERIMENT_ID
        validation_mask = df_final['experiment'] == self.config.VALIDATION_EXPERIMENT_ID
        train_metadata = df_final[~test_mask & ~validation_mask]
        test_metadata = df_final[test_mask]
        validation_metadata = df_final[validation_mask]

        train_df = get_merged_data(train_metadata)
        test_df = get_merged_data(test_metadata)
        validation_df = get_merged_data(validation_metadata)

        train_df = clean(train_df)
        test_df = clean(test_df)
        validation_df = clean(validation_df)

        if self.config.USE_SUPERCLASSES:
            print("---TRAINING WITH SUPERCLASSES---")
            unique_original_classes = set(train_df['class'].unique())
            mapped_classes = set(self.config.SUPERCLASS_MAPPING.keys())
            unmapped_classes = unique_original_classes - mapped_classes

            if unmapped_classes:
                print("Following classes are unmapped and dropped: ")
                for cls_name in sorted(list(unmapped_classes)):
                    print(f" -- {cls_name}")
            else:
                print("All classes are mapped successfully")
                
            train_df['class'] = train_df['class'].map(self.config.SUPERCLASS_MAPPING)
            test_df['class'] = test_df['class'].map(self.config.SUPERCLASS_MAPPING)
            validation_df['class'] = validation_df['class'].map(self.config.SUPERCLASS_MAPPING)

            train_df.dropna(subset=['class'], inplace=True)
            test_df.dropna(subset=['class'], inplace=True)
            validation_df.dropna(subset=['class'], inplace=True)

        self.le = LabelEncoder()
        train_df['class'] = self.le.fit_transform(train_df['class'])
        test_df['class'] = self.le.transform(test_df['class'])
        validation_df['class'] = self.le.transform(validation_df['class'])
        
        le_path = f"label_encoder_merged.joblib"
        joblib.dump(self.le, le_path)
        
        test_df_path = f"test_data_merged.pkl"
        test_df.to_pickle(test_df_path)

        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        train_df[self.config.SENSOR_COLS] = self.scaler.fit_transform(train_df[self.config.SENSOR_COLS])
        validation_df[self.config.SENSOR_COLS] = self.scaler.transform(validation_df[self.config.SENSOR_COLS])
        test_df[self.config.SENSOR_COLS] = self.scaler.transform(test_df[self.config.SENSOR_COLS])

        X_train_to_resample = train_df[self.config.SENSOR_COLS].values
        y_train_to_resample = train_df['class'].values

        if self.config.USE_OVERSAMPLING and self.config.RESAMPLING_STRATEGY != "NONE":
            counts_before = train_df['class'].value_counts()
            k_smote = min(5, min(counts_before.loc[counts_before > 0]) - 1)

            majority_count = counts_before.max()
            sampling_strategy_dict = counts_before.to_dict()
            for class_label, count in sampling_strategy_dict.items():
                if count < majority_count:
                    new_count = int(count * (1 + (self.config.MINORITY_INCREASE_PERCENTAGE / 100.0)))
                    new_count = min(new_count, majority_count)
                    sampling_strategy_dict[class_label] = new_count

            if k_smote < 1:
                print(f"Warning: Cannot apply {self.config.self.config.RESAMPLING_STRATEGY}")
                final_train_df = train_df
            else:
                sampler = None
                strategy_name = self.config.RESAMPLING_STRATEGY

                if self.config.RESAMPLING_STRATEGY == "SMOTE":
                    print(f"Applying SMOTE with target strategy: {sampling_strategy_dict}")
                    sampler = SMOTE(sampling_strategy=sampling_strategy_dict, k_neighbors=k_smote,
                                    random_state=self.config.RANDOM_STATE)
                elif self.config.RESAMPLING_STRATEGY == "SMOTEENN":
                    print(f"Applying SMOTEENN with target strategy: {sampling_strategy_dict}")
                    smote_component = SMOTE(sampling_strategy=sampling_strategy_dict, k_neighbors=k_smote,
                                            random_state=self.config.RANDOM_STATE)
                    sampler = SMOTEENN(smote=smote_component, random_state=self.config.RANDOM_STATE)
                elif self.config.RESAMPLING_STRATEGY == "ADASYN":
                    print(f"Applying ADASYN with target strategy: {sampling_strategy_dict}")
                    sampler = ADASYN(sampling_strategy=sampling_strategy_dict, n_neighbors=k_smote,
                                     random_state=self.config.RANDOM_STATE)
                elif self.config.RESAMPLING_STRATEGY == "SVMSMOTE":
                    sampler = SVMSMOTE(sampling_strategy=sampling_strategy_dict, k_neighbors=k_smote,
                                       random_state=self.config.RANDOM_STATE)
                elif self.config.RESAMPLING_STRATEGY == "UNDERSAMPLE":
                    print("Applying RandomUnderSampler...")
                    sampler = RandomUnderSampler(random_state=self.config.RANDOM_STATE)
                    strategy_name_for_plot = "RandomUnderSampler"

                if sampler:
                    try:
                        print(f"Applying {strategy_name}...")
                        X_resampled, y_resampled = sampler.fit_resample(X_train_to_resample, y_train_to_resample)
                        print(f"{strategy_name} applied. New training set size: {len(X_resampled)}")
                        final_train_df = pd.DataFrame(X_resampled, columns=self.config.SENSOR_COLS)
                        final_train_df['class'] = y_resampled
                    except Exception as e:
                        print(f"Error during {strategy_name}: {e}. Skipping resampling.")
                        final_train_df = train_df
                else:
                    print(f"Warning: Unknown self.config.RESAMPLING_STRATEGY. Skipping resampling.")
                    final_train_df = train_df
        else:
            print("No oversampling is used.")
            final_train_df = train_df

        class_counts = final_train_df['class'].value_counts()
        full_class_index = range(len(self.le.classes_))
        full_class_counts = pd.Series(0, index=full_class_index)
        full_class_counts.update(class_counts)
        class_weights = 1.0 / (full_class_counts + 1e-9)
        class_weights = class_weights / class_weights.sum() * len(self.le.classes_)
        class_weights_tensor = torch.tensor(class_weights.values, dtype=torch.float)

        train_ds = SensorDataset(final_train_df, self.config.SEQ_LEN, self.config.SENSOR_COLS, self.config.AUGMENT_TRAINING_DATA, self.config.AUGMENTATION_PROBABILITY)
        val_ds = SensorDataset(validation_df, self.config.SEQ_LEN, self.config.SENSOR_COLS, False)
        test_ds = SensorDataset(test_df, self.config.SEQ_LEN, self.config.SENSOR_COLS, False)

        print("Size of Training Set:", len(train_ds))
        print("Size of Validation Set:", len(val_ds))
        print("Size of Test Set:", len(test_ds))


        train_loader = DataLoader(train_ds, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        print("Data preparation complete.")
        return train_loader, val_loader, test_loader, self.le, class_weights_tensor