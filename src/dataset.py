import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import pickle
import config

class LoanApprovalDataset:
    def __init__(self):
        """Initialize dataset handler and internal state."""
        self.feature_transformer = None

        # Numeric and categorical columns based on your CSV
        self.numeric_cols = [
            "person_age",
            "person_income",
            "person_emp_exp",
            "loan_amnt",
            "loan_int_rate",
            "loan_percent_income",
            "cb_person_cred_hist_length",
            "credit_score"
        ]
        self.categorical_cols = [
            "person_gender",
            "person_education",
            "person_home_ownership",
            "loan_intent",
            "previous_loan_defaults_on_file"
        ]

    # ----------------- Public Methods -----------------
    def get_input_dim(self, data_loader):
        sample_X, _ = next(iter(data_loader))
        input_dim = sample_X.shape[1]
        print(f"• Input dimension: {input_dim}")
        return input_dim

    def prepare_data_for_training(self):
        """Prepare training, validation, and test DataLoaders from CSV."""
        df = self._load_csv()
        X = self._fit_feature_transformer(df)
        y = torch.tensor(df[["loan_status"]].values.astype(np.float32), dtype=torch.float32).reshape(-1, 1)

        X_train, X_val, X_test, y_train, y_val, y_test = self._split_dataset(X, y)
        train_loader, val_loader, test_loader = self._create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test)
        return train_loader, val_loader, test_loader

    def prepare_data_for_inference(self, df):
        """Prepare feature tensor for inference from new data."""
        X = self._transform_features(df)
        return X

    # ----------------- Save / Load -----------------
    def save_feature_transformer(self):
        os.makedirs(os.path.dirname(config.FEATURE_TRANSFORMER_PATH), exist_ok=True)
        with open(config.FEATURE_TRANSFORMER_PATH, "wb") as f:
            pickle.dump(self.feature_transformer, f)
        print(f"• Feature transformer saved to {config.FEATURE_TRANSFORMER_PATH}")

    def load_feature_transformer(self):
        with open(config.FEATURE_TRANSFORMER_PATH, "rb") as f:
            self.feature_transformer = pickle.load(f)
        print(f"• Feature transformer loaded from {config.FEATURE_TRANSFORMER_PATH}")

    # ----------------- Private Helpers -----------------

    def _load_csv(self):
        """Load CSV into a pandas DataFrame."""
        df = pd.read_csv(config.DATASET_CSV_PATH)
        print(f"• Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _fit_feature_transformer(self, df):
        self.feature_transformer = ColumnTransformer(transformers=[
            ("num", StandardScaler(), self.numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), self.categorical_cols)
        ])
        X = self.feature_transformer.fit_transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.tensor(X, dtype=torch.float32)

    def _transform_features(self, df):
        X = self.feature_transformer.transform(df)
        if hasattr(X, "toarray"):
            X = X.toarray()
        return torch.tensor(X, dtype=torch.float32)

    def _split_dataset(self, X, y):
        """Split dataset into training, validation, and test subsets."""

        if not config.SPLIT_DATASET:
            print("• Dataset splitting disabled → using same data for train/val/test.")
            return X, X, X, y, y, y

        dataset = TensorDataset(X, y)
        n_total = len(dataset)
        n_train = int(config.TRAIN_SPLIT_RATIO * n_total)
        n_val = int(config.VAL_SPLIT_RATIO * n_total)
        n_test = n_total - n_train - n_val

        generator = (
            torch.Generator().manual_seed(config.SPLIT_RANDOMIZATION_SEED)
            if config.SPLIT_RANDOMIZATION_SEED is not None else None
        )

        train_ds, val_ds, test_ds = random_split(
            dataset, [n_train, n_val, n_test], 
            generator=generator
        )

        X_train, y_train = train_ds[:][0], train_ds[:][1]
        X_val, y_val = val_ds[:][0], val_ds[:][1]
        X_test, y_test = test_ds[:][0], test_ds[:][1]

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Return train, val, test DataLoaders."""
        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=config.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=config.BATCH_SIZE, shuffle=False)
        return train_loader, val_loader, test_loader
