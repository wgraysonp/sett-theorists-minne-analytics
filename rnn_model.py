import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import os
import argparse
import random

DATA_DIR = os.path.join(os.getcwd(), 'data')



def get_parser():
    parser = argparse.ArgumentParser(description='Minne 2025 RNN Model')
    parser.add_argument('--lr_max', default=1e-3, type=float, help='max learning rate for training')
    parser.add_argument('--lr_min', default=1e-4, type=float, help='min learning rate for training')
    parser.add_argument('--n_epochs', default=20, type=int, help='number of epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--embed_dim', default=384, type=int, help='embedding dimesion for sentiment analysis')
    parser.add_argument('--hidden_dim', default=128, type=int, help='dimension of hidden layers')
    parser.add_argument('--feature_dim', default=2, type=int, help='additional numeric features')
    parser.add_argument('--t0', default=10, type=int, help='number of epochs until restart for lr schedule')
    return parser


# Load data
def load_data(random_drop=True):
    df = pd.read_csv(DATA_DIR + '/updated_mathclength_sorted_Training.csv', low_memory=False)
    df = df.dropna(subset=['Completion Date', 'Match Support Contact Notes'])
    df['Completion Date'] = pd.to_datetime(df['Completion Date'])
    static_columns = [
    'Big Age', 
    'Big Gender', 
    'Big Race/Ethnicity',
    'Little Gender', 
    'Little Participant: Race/Ethnicity',
    'Program', 
    'Program Type',
    ]
    static_features = []
    cat_cols = [col for col in static_columns if df[col].dtype == 'object']
    num_cols = [col for col in static_columns if col not in cat_cols]
    for col in cat_cols:
        df[col] = df[col].fillna('unknown')
        encoder = OneHotEncoder(sparse=False)
        cat_encoded = encoder.fit_transform(df[[col]])
        #df_encoded = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names(), index=df.index)
        df_encoded = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out([col]), index=df.index)
        df = pd.concat([df, df_encoded], axis=1).drop(columns=[col], axis=1)
        #static_features = static_features + encoder.get_feature_names_out().tolist()
        static_features += feature_names.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].mean())
        static_features.append(col)

   # label_encoders = {}
   # for col in static_columns:
   #     if df[col].dtype == 'object':
   #         le = LabelEncoder()
   #         df[col] = df[col].fillna('UnKnown')
   #         df[col] = le.fit_transform(df[col])
   #         label_encoders[col] = le
   #     else:
   #         df[col] = df[col].fillna(df[col].mean())

    # Group and sort by Match ID and Completion Date
    grouped = df.groupby('Match ID 18Char')

    # Prepare data tuples: (sequence of notes, numerical features, target final match length)
    data = []
    for match_id, group in grouped:
        group_sorted = group.sort_values(by='Completion Date')
        final_match_length = group_sorted['Match Length'].iloc[-1] # Target is match length
        if random_drop and len(group_sorted) > 1:
            drop_idx = random.randint(len(group_sorted)//2, len(group_sorted))
            group_sorted = group_sorted.iloc[:drop_idx]

        notes_sequence = group_sorted['Match Support Contact Notes'].tolist()
        
        # time dependent features
        #avg_match_length = group_sorted['Match Length'].mean()
        current_match_length = (group_sorted['Completion Date'].iloc[-1] - group_sorted['Completion Date'].iloc[0]).days
        num_contacts = len(group_sorted)

        # static features
        static_values = group_sorted.iloc[0][static_features].values.astype(float)

        combined_features = [num_contacts, current_match_length] + static_values.tolist()
        #final_match_length = group_sorted['Match Length'].iloc[-1]  # Target is last match length

        data.append((notes_sequence, combined_features, final_match_length))

    return data, len(static_features)

# Custom Dataset
class MatchDataset(Dataset):
    def __init__(self, data, random_drop=False):
        self.data = data
        self.random_drop = random_drop
        # Initialize SentenceTransformer model for text embedding
        self.sbert = SentenceTransformer('all-MiniLM-L6-v2')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        notes, features, target = self.data[idx]
        #if self.random_drop and len(notes) > 1:
        #    total_contacts = len(notes)
        #    drop_idx = random.randint(total_contacts//2, total_contacts)
        #    notes = notes[:drop_idx]
        #    #print(len(notes))
        #    assert len(notes) > 0, "empty notes"
        #    features[0] = len(notes)
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
        self.fc_1 = nn.Linear(hidden_dim + feature_dim, 128)
        self.fc_2 = nn.Linear(128, 1)

    def forward(self, x, lengths, features):
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input)
        combined = torch.cat([hidden[-1], features], dim=1)
        output = self.fc_1(combined)
        output = F.relu(output)
        output = self.fc_2(output)
        return output.squeeze()
    

# Training loop
def train(model, device, optimizer, criterion, loader, scheduler, epoch):
    model.train()
    iters = len(loader)
    epoch_loss = 0
    preds_list = []
    targets_list = []
    for i, (padded_seqs, lengths, features, targets) in enumerate(tqdm(loader)):
        padded_seqs, lengths, features, targets = padded_seqs.to(device), lengths.to(device), features.to(device), targets.to(device)
        optimizer.zero_grad()
        preds = model(padded_seqs, lengths, features)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        scheduler.step(epoch + i/iters)

        epoch_loss += loss.item()
        preds_list.append(preds.cpu().detach().numpy())
        targets_list.append(targets.cpu().numpy())

        # Calculate RMSE at the end of the epoch
    preds_all = np.concatenate(preds_list)
    targets_all = np.concatenate(targets_list)
    rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2)/300)

    print(f"Epoch {epoch+1}, Train Loss: {epoch_loss / len(loader):.4f}, Train RMSE: {rmse:.4f}")
    print('epoch={}, learning rate={:.4f}'.format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    return rmse

def test(model, device, loader, epoch):
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
    test_rmse = np.sqrt(np.mean((preds_all - targets_all) ** 2)/300)
    print(f"Epoch {epoch+1}, Test RMSE: {test_rmse:.4f}")
    return test_rmse

def main():
    parser = get_parser()
    args = parser.parse_args()

    data, n_static = load_data(random_drop=True)

    # Hyperparameters
    embed_dim = args.embed_dim  # Embedding size of 'all-MiniLM-L6-v2'
    hidden_dim = args.hidden_dim
    feature_dim = args.feature_dim + n_static # Number of additional numeric features
    #feature_dim = 126 # hard coded this. The number n_static returned by load_data is wrong. Not sure why. 
    batch_size = args.batch_size

    print(feature_dim)


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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr_max)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=args.t0, eta_min=args.lr_min)

    # Run on GPU if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    #if device == 'cuda':
    #    model = torch.nn.DataParallel(model)
    #    cudnn.benchmark = True

    train_mses = []
    test_mses = []

    for epoch in range(args.n_epochs):
        train_mse = train(model, device, optimizer, criterion, train_loader, scheduler, epoch)
        test_mse = test(model, device, test_loader, epoch)
        train_mses.append(train_mse)
        test_mses.append(test_mse)

    #train(model, device, optimizer, criterion, train_loader, lr=args.lr, num_epochs=args.n_epochs)
    #test(model, device, test_loader)

    torch.save(model.state_dict(), 'saved_models/rnn_model_state_dict.pth')
    if not os.path.isdir('curve'):
        os.mkdir('curve')
    torch.save({'train_rmse': train_mses, 'test_rmse': test_mses}, os.path.join('curve', 'training_curve'))

if __name__ == "__main__":
    main()
