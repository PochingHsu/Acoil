import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
seed = 42
torch.manual_seed(seed)


# ANN model
class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = layer_dim
        self.output_dim = output_dim
        # self.dropout = dropout_prob
        self.flatten = nn.Flatten()
        self.layers = nn.ModuleDict()
        self.layers["input"] = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        for i in range(self.n_layers):
            self.layers[f"hidden_{i}"] = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            # self.layers[f"dropout{i}"] = nn.Dropout(dropout_prob)
        self.layers["output"] = nn.Linear(in_features=hidden_dim, out_features=output_dim)

    def forward(self, x):
        x = self.layers["input"](self.flatten(x))
        for i in range(self.n_layers):
            x = F.relu(self.layers[f"hidden_{i}"](x))
            # x = self.layers[f"dropout{i}"](x)
        return self.layers["output"](x)


class Optimization:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

    def train_step(self, x, y):
        # Sets model to train mode
        self.model.train()

        # Makes predictions
        yhat = self.model(x)

        # Computes loss
        loss = self.loss_fn(y, yhat)

        # Computes gradients
        loss.backward()

        # Updates parameters and zeroes gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Returns the loss
        return loss.item()

    def train(self, train_loader, val_loader, batch_size, n_epochs, n_features):
        model_path = f'models/{self.model}_{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # device = torch.device("cpu")
        early_stopper = EarlyStopper(patience=20, min_delta=1e-6)
        for epoch in range(1, n_epochs + 1):
            batch_losses = []
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.view([batch_size, -1, n_features]).to(device)
                y_batch = y_batch.to(device)
                loss = self.train_step(x_batch, y_batch)
                batch_losses.append(loss)
            training_loss = np.mean(batch_losses)
            self.train_losses.append(training_loss)

            with torch.no_grad():
                batch_val_losses = []
                for x_val, y_val in val_loader:
                    x_val = x_val.view([batch_size, -1, n_features]).to(device)
                    y_val = y_val.to(device)
                    self.model.eval()
                    yhat = self.model(x_val)
                    val_loss = self.loss_fn(y_val, yhat).item()
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.val_losses.append(validation_loss)

                # if (epoch <= 10) | (epoch % 50 == 0):
                print(
                    f"[{epoch}/{n_epochs}] Training loss: {training_loss:.6f}\t Validation loss: {validation_loss:.6f}")
            val_losses_mean = np.mean(self.val_losses[-20:])
            # early stop
            if epoch >= 20:
                if early_stopper.early_stop(val_losses_mean):
                    print('Early stopping!')
                    break
        # torch.save(self.model.state_dict(), model_path)

    def evaluate(self, test_loader, batch_size, n_features):
        with torch.no_grad():
            predictions = []
            values = []
            for x_test, y_test in test_loader:
                x_test = x_test.view([batch_size, -1, n_features]).to(device)
                y_test = y_test.to(device)
                self.model.eval()
                yhat = self.model(x_test)
                predictions.append(yhat.to(device).detach().cpu().numpy())
                values.append(y_test.to(device).detach().cpu().numpy())

        return predictions, values

    def plot_losses(self):
        plt.plot(self.train_losses, label="Training loss")
        plt.plot(self.val_losses, label="Validation loss")
        plt.legend()
        plt.title("Losses")
        plt.show()
        plt.close()


class EarlyStopper:
    def __init__(self, patience=20, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, val_losses_mean):
        if val_losses_mean < self.min_validation_loss:
            self.min_validation_loss = val_losses_mean
            self.counter = 0
        elif val_losses_mean > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class HyperparametersOP:
    def __init__(self, input_dim, output_dim, dataset, k_folds, n_epochs, device):
        self.learning_rate = 1e-3
        self.weight_decay = 1e-6
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dataset = dataset
        self.k_folds = k_folds
        self.n_epochs = n_epochs
        self.device = device

    def ann_bo(self, params_nn):
        # Define the K-fold Cross Validator
        kfold = KFold(n_splits=self.k_folds, shuffle=True)
        results = {}
        for fold, (train_ids, test_ids) in enumerate(kfold.split(self.dataset)):
            print(f'FOLD {fold}')
            train_subsampler = SubsetRandomSampler(train_ids)
            test_subsampler = SubsetRandomSampler(test_ids)
            seed = 42
            torch.manual_seed(seed)
            batch_size_power = params_nn['batch_size_power']
            layer_dim = params_nn['layer_dim']
            hidden_size_power = params_nn['num_hidden_units']
            hidden_dim = 2 ** round(hidden_size_power)
            batch_size = 2 ** round(batch_size_power)
            train_loader = DataLoader(self.dataset, batch_size, sampler=train_subsampler, drop_last=True)
            val_loader = DataLoader(self.dataset, batch_size, sampler=test_subsampler, drop_last=True)
            val_loader_one = DataLoader(self.dataset, 1, shuffle=False, drop_last=True)
            model_params = {'input_dim': self.input_dim,
                            'hidden_dim': round(hidden_dim),
                            'layer_dim': round(layer_dim),
                            'output_dim': self.output_dim}
            model = get_model('ann', model_params)
            model = model.to(self.device)
            loss_fn = nn.MSELoss(reduction="mean")
            optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            opt = Optimization(model=model, loss_fn=loss_fn, optimizer=optimizer)
            # Training
            opt.train(train_loader, val_loader, batch_size=batch_size, n_epochs=self.n_epochs, n_features=self.input_dim)
            # opt.plot_losses()
            predictions, values = opt.evaluate(val_loader_one, batch_size=1, n_features=self.input_dim)
            preds = np.asarray(predictions)
            vals = np.asarray(values)
            preds = preds[1, :, :]  # Get the last hidden unit prediction
            vals = vals[1, :, :]  # Get the last value of time series data
            # error
            mae = mean_absolute_error(preds, vals)
            #rms = mean_squared_error(preds, vals, squared=False)
            #neg_cvrmse = -rms / np.mean(vals)
            #cvrmse = rms / np.mean(vals)
            results[fold] = mae
        error_mean = np.mean(list(results.values()))
        return error_mean


def get_model(model, model_params):
    models = {
        "ann": NeuralNetwork
    }
    return models.get(model.lower())(**model_params)


def inverse_transform(scaler, df, columns):
    for col in columns:
        df[col] = scaler.inverse_transform(df[col])
    return df


def format_predictions(predictions, values, df_test, scaler):
    vals = np.concatenate(values, axis=0).ravel()
    preds = np.concatenate(predictions, axis=0).ravel()
    df_result = pd.DataFrame(data={"value": vals, "prediction": preds}, index=df_test.head(len(vals)).index)
    df_result = df_result.sort_index()
    df_result = inverse_transform(scaler, df_result, [["value", "prediction"]])
    return df_result
