import pytorch_lightning as pl
import torchmetrics
import torch
import torch.nn as nn

class LanguageModel(pl.LightningModule):
    def __init__(self, hidden_dim, n_layers, len_alphabet, len_molecule, dropout = 0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.len_alphabet = len_alphabet
        self.len_molecule = len_molecule
        self.dropout = dropout

        self.lstm = nn.LSTM(
            self.len_alphabet, 
            hidden_dim,
            num_layers = n_layers,
            dropout = dropout
        )
        self.act = nn.ReLU()
        self.linear = nn.Linear(hidden_dim, self.len_alphabet)

        self.loss_fn = nn.CrossEntropyLoss()    # assume logit input
        self.train_vae_acc = torchmetrics.Accuracy()
        self.val_vae_acc = torchmetrics.Accuracy()

    def forward(self, x, hidden = None):
        # x = [batch_size, len_molecule, len_alphabet]
        output = []
        for i in range(self.len_molecule):
            out, hidden = self.lstm(x[:, i, :].squeeze().unsqueeze(0), hidden)
            out = out.squeeze()
            out = self.act(out)
            out = self.linear(out)
            output.append(out)

        output = torch.stack(output, dim=1)
        return output

    def training_step(self, batch, batch_idx):
        batch = batch.type(torch.FloatTensor)
        pred = self(batch)
        pred = pred.view(-1, pred.size(-1))     # [batch_size * len_alphabet, len_molecule]
        labels = batch.argmax(dim=-1).view(-1)  # [batch_size * len_alphbaet * len_molecule]
        loss = self.loss_fn(pred, labels)

        pred = pred.softmax(dim=-1)
        self.train_vae_acc(pred, labels)

        self.log(f'train_accuracy', self.train_vae_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.type(torch.FloatTensor)
        pred = self(batch)
        pred = pred.view(-1, pred.size(-1))     # [batch_size * len_alphabet, len_molecule]
        labels = batch.argmax(dim=-1).view(-1)  # [batch_size * len_alphbaet * len_molecule]
        loss = self.loss_fn(pred, labels)

        pred = pred.softmax(dim=-1)
        self.val_vae_acc(pred, labels)
        
        self.log(f'val_accuracy', self.val_vae_acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def sample(self, num_chars, starting_chars = 5):
        return

    