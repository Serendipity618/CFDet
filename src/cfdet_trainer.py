import numpy as np
import torch
import os
from sklearn import metrics
from torch.autograd import Variable
from collections import deque
from torch import optim, nn


class CFDetTrainer:
    def __init__(self, cfdet, ad_model, train_test_loader, batch_size_train_test, epochs, center, device,
                 baseline_sequence, triplet_lambda, sparsity_lambda, continuity_lambda):
        self.model = cfdet
        self.ad_model = ad_model

        self.train_test_loader = train_test_loader
        self.batch_size_train_test = batch_size_train_test

        self.optimiser = optim.Adam(self.model.generator.parameters(), lr=1e-3)
        self.mse_criterion = nn.MSELoss()
        self.tml_criterion = nn.TripletMarginLoss(margin=1, reduction='none')

        self.epochs = epochs
        self.triplet_lambda = triplet_lambda
        self.sparsity_lambda = sparsity_lambda
        self.continuity_lambda = continuity_lambda
        self.center_batch = torch.repeat_interleave(torch.unsqueeze(center, 0), batch_size_train_test, dim=0).to(device)
        self.baseline_sequence = baseline_sequence

        self.device = device

    def train(self, output_path):
        """Train CFDet model using reinforcement learning."""
        total_loss_list, continuity_loss_list, sparsity_loss_list = [], [], []
        distance_loss_list, reward_list, loss_list = [], [], []
        min_loss = float('inf')

        for epoch in range(self.epochs):
            z_history_rewards = deque(maxlen=200)
            z_history_rewards.append(0.0)

            epoch_distance_loss, epoch_continuity_loss = [], []
            epoch_sparsity_loss, epoch_rl_loss = [], []
            epoch_reward, epoch_loss = [], []

            for param in self.ad_model.parameters():
                param.requires_grad = False

            for sequence, sequence_label, _ in self.train_test_loader:
                sequence = sequence.to(self.device)
                baseline = Variable(torch.FloatTensor([float(np.mean(z_history_rewards))]))

                self.optimiser.zero_grad()
                z, neg_log_probs = self.model.generate(sequence)
                distance_loss, rl_loss, rewards, continuity_loss, sparsity_loss, _ = (
                    self.get_loss(sequence, z, neg_log_probs, baseline, self.ad_model, self.baseline_sequence)
                )
                rl_loss.backward()  # Compute gradients
                self.optimiser.step()  # Update model parameters

                epoch_distance_loss.append(torch.mean(distance_loss).item())
                epoch_continuity_loss.append(torch.mean(continuity_loss).item())
                epoch_sparsity_loss.append(torch.mean(sparsity_loss).item())
                epoch_rl_loss.append(rl_loss.item())
                epoch_reward.append(torch.sum(rewards).item())
                epoch_loss.append(torch.sum(-rewards).item())
                z_history_rewards.append(np.mean(rewards.cpu().data.numpy()))

        current_loss = (np.mean(epoch_distance_loss) + self.continuity_lambda * np.mean(epoch_continuity_loss) +
                        self.sparsity_lambda * np.mean(epoch_sparsity_loss))

        if current_loss < min_loss:
            min_loss = current_loss
            torch.save(self.model.generator.state_dict(), os.path.join(output_path, "state_dict_minloss.bin"))

        torch.save(self.model.generator.state_dict(), os.path.join(output_path, "state_dict_final.bin"))

    def evaluate(self, val_loader, output_path):
        """Evaluate the CFDet model."""
        y_key_truth, y_key_pred = [], []

        self.model.eval()
        self.ad_model.eval()

        with torch.no_grad():
            for sequence, sequence_label, key_label in val_loader:
                sequence = sequence.to(self.device)
                z_out, _ = self.model.generate(sequence, training=False)

                key_label_list = key_label.tolist()
                z_list = z_out.data.tolist()

                for i in range(len(sequence_label)):
                    y_key_truth += key_label_list[i]
                    y_key_pred += z_list[i]

        report = metrics.classification_report(y_key_truth, y_key_pred, digits=4)
        cm = metrics.confusion_matrix(y_key_truth, y_key_pred)
        fpr, tpr, _ = metrics.roc_curve(y_key_truth, y_key_pred, pos_label=1)
        auc = metrics.auc(fpr, tpr)

        print(report)
        print(cm)
        print(auc)

        with open(output_path, 'w') as f:
            f.write(report + '\n')
            f.write(str(cm) + '\n')
            f.write(str(auc) + '\n')
            f.write('-' * 50 + '\n')

    def get_loss(self, x, z, neg_log_probs, average_reward, model, baseline_sequence, sequence_length=20.0):
        """
        Computes loss, including continuity, sparsity, and distance loss.

        Args:
            x (Tensor): Input sequence tensor.
            z (Tensor): Generated counterfactual mask.
            neg_log_probs (Tensor): Negative log probabilities.
            average_reward (Tensor): Running average of rewards.
            batch_size (int): Batch size for training.
            model (nn.Module): The anomaly detection model.
            sequence_length (float, optional): Length of input sequences.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
                - Distance loss
                - Reinforcement learning loss
                - Rewards
                - Continuity loss
                - Sparsity loss
                - Advantage values for RL training
        """
        # Compute continuity loss: Encourages smooth counterfactual modifications
        z_ = torch.cat([z[:, 1:], z[:, -1:]], dim=-1)  # Shift z for continuity check
        continuity_ratio = torch.sum(torch.abs(z - z_), dim=-1) / sequence_length
        percentage = (self.model.count_pieces - 1) / sequence_length
        continuity_loss = torch.abs(continuity_ratio - percentage)

        # Compute sparsity loss: Encourages minimal modifications
        sparsity_ratio = torch.sum(z, dim=-1) / sequence_length
        percentage = self.model.count_tokens / sequence_length
        sparsity_loss = torch.abs(sparsity_ratio - percentage)

        # Generate counterfactual sequences
        anomalous_entry = x * z + baseline_sequence * (1 - z)
        anti = x * (1 - z) + baseline_sequence * z

        # Compute hidden representations
        hidden_anomalous_entry = model(anomalous_entry)
        hidden_anti = model(anti)

        # Compute triplet loss for counterfactual separation
        distance_loss = (
                self.tml_criterion(self.center_batch, hidden_anti, hidden_anomalous_entry)
                + self.mse_criterion(self.center_batch, hidden_anti)
                - self.mse_criterion(self.center_batch, hidden_anomalous_entry)
        )

        # Compute rewards for RL training
        average_reward = average_reward.to(self.device)
        rewards = -(
                self.triplet_lambda * distance_loss
                + self.sparsity_lambda * sparsity_loss
                + self.continuity_lambda * continuity_loss
        ).detach()

        # Compute RL advantages
        advantages = rewards - average_reward  # (batch_size,)
        advantages_expand_ = advantages.unsqueeze(-1).expand_as(neg_log_probs)

        # Compute RL loss
        rl_loss = torch.sum(neg_log_probs * advantages_expand_)

        return distance_loss, rl_loss, rewards, continuity_loss, sparsity_loss, advantages_expand_
