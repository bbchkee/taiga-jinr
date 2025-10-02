
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class GammaShowerRegressor:
    def __init__(self, model_path='gamma_regressor.pth'):
        self.model_path = model_path
        self.model = self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=3e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10, verbose=True)
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.df = None
        self.test_indices = None
        self.train_loader = None
        self.test_loader = None

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def load_data(self, df):
        df = df[['size', 'dist[0]', 'width[0]', 'length[0]', 'numb_pix', 'energy', 'reconstructed_energy']].dropna()
        self.df = df
        X = df[['size', 'dist[0]', 'width[0]', 'length[0]', 'numb_pix']].values.astype(np.float32)
        y = df['energy'].values.astype(np.float32).reshape(-1, 1)
        X = self.scaler_X.fit_transform(X)
        y = self.scaler_y.fit_transform(y)
        X_train, X_test, y_train, y_test, _, test_idx = train_test_split(X, y, df.index, test_size=0.2, random_state=42)
        self.test_indices = test_idx
        self.train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=32, shuffle=True)
        self.test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=32, shuffle=False)

    def train(self, epochs=200):
        best = float('inf')
        patience = 5
        bad = 0
        self.model.train()
        for ep in range(epochs):
            total = 0.0
            for Xb, yb in self.train_loader:
                Xb, yb = Xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                out = self.model(Xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                total += loss.item()
            avg = total / max(1, len(self.train_loader))
            if avg < best:
                best = avg; bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break
            self.scheduler.step(avg)

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)

    def plot_energy_histograms(self, out_dir='plots'):
        import os; os.makedirs(out_dir, exist_ok=True)
        self.model.eval()
        true_E, pred_E = [], []
        with torch.no_grad():
            for Xb, yb in self.test_loader:
                Xb = Xb.to(self.device)
                pred = self.model(Xb).cpu().numpy()
                pred = self.scaler_y.inverse_transform(pred)
                yb = self.scaler_y.inverse_transform(yb.cpu().numpy())
                true_E.extend(yb.flatten()); pred_E.extend(pred.flatten())

        bins = np.arange(0, 205, 5)
        fig = plt.figure(figsize=(8,6))
        sns.histplot(x=true_E, y=pred_E, bins=[bins, bins], cbar=True)
        plt.xlabel("True Energy"); plt.ylabel("Predicted Energy"); plt.xlim(0,200); plt.ylim(0,200); plt.grid()
        fig.tight_layout(); fig.savefig(f"{out_dir}/energy_diff_regressor.png", dpi=150); plt.close(fig)

        if self.df is not None and self.test_indices is not None:
            test_df = self.df.loc[self.test_indices]
            fig = plt.figure(figsize=(8,6))
            sns.histplot(x=test_df['energy'], y=test_df['reconstructed_energy'], bins=[bins, bins], cbar=True)
            plt.xlabel("True Energy"); plt.ylabel("Reconstructed Energy"); plt.xlim(0,200); plt.ylim(0,200); plt.grid()
            fig.tight_layout(); fig.savefig(f"{out_dir}/energy_diff_classic.png", dpi=150); plt.close(fig)

            err_reg = (np.array(pred_E) - np.array(test_df['energy']))/np.array(test_df['energy'])
            err_cls = (np.array(test_df['reconstructed_energy']) - np.array(test_df['energy']))/np.array(test_df['energy'])
            fig = plt.figure(figsize=(8,6))
            sns.histplot(err_reg, bins=50, label='Regressor', alpha=0.6)
            sns.histplot(err_cls, bins=50, label='Classic', alpha=0.6)
            plt.xlabel("Relative Energy Residual"); plt.ylabel("Frequency"); plt.legend(); plt.grid()
            fig.tight_layout(); fig.savefig(f"{out_dir}/residuals_comparison.png", dpi=150); plt.close(fig)
