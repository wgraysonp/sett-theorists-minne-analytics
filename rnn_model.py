import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
import os
import argparse

DATA_DIR = os.path.join(os.getcwd(), 'data')



def get_parser():
    parser = argparse.ArgumentParser(description='Minne 2025 RNN Model')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate for training')
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs to run')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dimesion for sentiment analysis')
    parser.add_argument('--hidden_dim', default=128, type=int, help='dimension of hidden layers')
    parser.add_argument('--feature_dim', default=2, type=int, help='additional numeric features')
    return parser


# Load data
def load_data():
    df = pd.read_csv(DATA_DIR + '/updated_mathclength_sorted_Training.csv', low_memory=False)
    df = df.dropna(subset=['Completion Date', 'Match Support Contact Notes'])
    df['Completion Date'] = pd.to_datetime(df['Completion Date'])

    # Group and sort by Match ID and Completion Date
    grouped = df.groupby('Match ID 18Char')

    # Prepare data tuples: (sequence of notes, numerical features, target final match length)
    data = []
    for match_id, group in grouped:
        group_sorted = group.sort_values(by='Completion Date')
        notes_sequence = group_sorted['Match Support Contact Notes'].tolist()
        # Example of adding numerical features - you can customize this
        avg_match_length = group_sorted['Match Length'].mean()
        num_contacts = len(group_sorted)
        final_match_length = group_sorted['Match Length'].iloc[-1]  # Target is last match length
        data.append((notes_sequence, [avg_match_length, num_contacts], final_match_length))

    return data

# Custom Dataset
class MatchDataset(Dataset):
    def __init__(self, data):
        self.data = data
        # Initialize SentenceTransformer model for text embedding
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        notes, features, target = self.data[idx]
        embeddings = self.sbert.encode(notes)  # Shape: (seq_len, embed_dim)
        features = torch.tensor(features, dtype=torch.float32)
        return torch.tensor(embeddings, dtype=torch.float32), features, torch.tensor(target, dtype=torch.float32)

# Collate function to handle variable sequence lengths
def collate_fn(batch):
    sequences, features, targets = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    features = torch.stack(features)
    return padded_sequences, torch.tensor(lengths), features, torch.tensor(targets)

# Model definition
class SentimentRNN(nn.Module):
    def __init__(self, embed_dim, hidden_dim, feature_dim):
        super(SentimentRNN, self).__init__()
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim + feature_dim, 1)

    def forward(self, x, lengths, features):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)
        combined = torch.cat([hidden[-1], features], dim=1)
        output = self.fc(combined)
        return output.squeeze()

# Training loop
def train(model, device, optimizer, criterion, loader, lr=1e-3, num_epochs=5):
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        preds_list = []
        targets_list = []
        for padded_seqs, lengths, features, targets in tqdm(loader):
            padded_seqs, lengths, features, targets = padded_seqs.to(device), lengths.to(device), features.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(padded_seqs, lengths, features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            preds_list.append(preds.cpu().detach().numpy())
            targets_list.append(targets.cpu().numpy())

        # Calculate RMSE at the end of the epoch
        preds_all = np.concatenate(preds_list)
        targets_all = np.concatenate(targets_list)
        rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss / len(loader):.4f}, Train RMSE: {rmse:.4f}")

def test(model, device, loader):
    # Evaluation on test set
    model.eval()
    preds_list = []
    targets_list = []
    with torch.no_grad():
        for padded_seqs, lengths, features, targets in loader:
            padded_seqs, lengths, features, targets = padded_seqs.to(device), lengths.to(device), features.to(device), targets.to(device)
            preds = model(padded_seqs, lengths, features)
            preds_list.append(preds.cpu().numpy())
            targets_list.append(targets.cpu().numpy())

    preds_all = np.concatenate(preds_list)
    targets_all = np.concatenate(targets_list)
    test_rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2))
    print(f"Test RMSE: {test_rmse:.4f}")

def main():
    parser = get_parser()
    args = parser.parse_args()

    # Hyperparameters
    embed_dim = args.embed_dim  # Embedding size of 'all-MiniLM-L6-v2'
    hidden_dim = args.hidden_dim
    feature_dim = args.feature_dim  # Number of additional numeric features
    batch_size = args.batch_size

    data = load_data()

    # Dataset and DataLoader
    dataset = MatchDataset(data)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, Loss, Optimizer
    model = SentimentRNN(embed_dim, hidden_dim, feature_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Run on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    train(model, device, optimizer, criterion, train_loader, lr=args.lr, num_epochs=args.n_epochs)
    test(model, device, test_loader)

    torch.save(model.state_dict(), 'rnn_model_state_dict.pth')

if __name__ == "__main__":
    main()
