import os
import torch
import numpy as np
from sklearn import metrics
from torch import nn, optim


class SVDDTrainer:
    def __init__(self, model, train_loader, device, batch_size=512, epochs=50):
        self.model = model
        self.train_loader = train_loader
        self.optimiser = optim.Adam(model.parameters(), lr=1e-3)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        self.hidden_dim = model.hidden_dim
        self.epochs = epochs
        self.center = None
        self.radius = None
        self.device = device

    def train(self, output_path):
        """Train the DeepSVDD model."""
        print("Anomaly detection model training started...")  # Indicate the start of training
        total_loss = []

        for epoch in range(self.epochs):
            epoch_loss = []
            hidden_sum = torch.zeros((self.batch_size, self.hidden_dim)).to(self.device)

            # Compute center during the first 20 epochs
            if epoch < 20:
                self.model.eval()
                with torch.no_grad():
                    for sequence, sequence_label, _ in self.train_loader:
                        sequence = sequence.to(self.device)
                        hidden_sum += self.model(sequence)

                # Compute center of the feature representations
                self.center = (torch.mean(hidden_sum, axis=0) / len(self.train_loader)).detach()
                center_batch = self.center.repeat(self.batch_size, 1)

            self.model.train()
            for sequence, sequence_label, _ in self.train_loader:
                sequence = sequence.to(self.device)
                self.optimiser.zero_grad()
                hidden = self.model(sequence)

                # Compute MSE loss between feature representations and center
                loss = self.criterion(hidden, center_batch.to(self.device))
                epoch_loss.append(loss.item())

                # Backpropagation and optimization step
                loss.backward()
                self.optimiser.step()

            print(f"Epoch {epoch + 1}, MSE: {np.mean(epoch_loss)}")  # Print loss per epoch
            total_loss.append(np.max(epoch_loss))

        # Store final radius based on the last epoch loss
        self.radius = total_loss[-1]

        # Save trained model
        torch.save(self.model.state_dict(), os.path.join(output_path, "DeepSVDD.bin"))

        print("Training completed successfully.")  # Indicate the successful completion of training
        return self.center.tolist(), self.radius

    def evaluate(self, data_loader, output_path):
        """Evaluate the DeepSVDD model."""
        self.model.eval()
        y_pred, y_truth = [], []

        with torch.no_grad():
            for sequence, sequence_label, _ in data_loader:
                y_truth.extend(sequence_label.tolist())
                sequence = sequence.to(self.device)
                hidden = self.model(sequence)
                distance = torch.mean(torch.square(hidden - self.center.to(self.device)), dim=1)
                y_pred.extend([int(i > self.radius) for i in distance])

        report = metrics.classification_report(y_truth, y_pred, digits=4)
        cm = metrics.confusion_matrix(y_truth, y_pred)
        fpr, tpr, _ = metrics.roc_curve(y_truth, y_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print(report)
        print(cm)
        print(auc)

        with open(output_path, 'w') as f:
            f.write(report + '\n')
            f.write(str(cm) + '\n')
            f.write(str(auc) + '\n')
            f.write('-' * 50 + '\n')
