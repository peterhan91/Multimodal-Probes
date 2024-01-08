import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import json
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import roc_auc_score

import clip
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Dataset

# Assumed device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CLIPImageDataset(Dataset):
    def __init__(self, list_of_images, preprocessing):
        self.images = list_of_images
        self.preprocessing = preprocessing

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.preprocessing(Image.open(self.images[idx]))  # preprocess from clip.load
        return images
    
@torch.no_grad()
def get_embs(loader, model, model_name):
    all_embs = []
    for images in tqdm(loader):
        images = images.to('cuda')
        if model_name in ["clip", "plip"]:
            all_embs.append(model.encode_image(images).cpu().numpy())
        else:
            all_embs.append(model(images).squeeze().cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)
    return all_embs

# Function to save the probe
def save_probe(probe, dataset, prompt_type, layer, base_path):
    save_path = os.path.join(base_path, dataset, prompt_type)
    os.makedirs(save_path, exist_ok=True)
    filename = f'model_layer_{layer}.pth'
    torch.save(probe.state_dict(), os.path.join(save_path, filename))
    return save_path

# Logistic Regression Model
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Function to train the model
def train(model, criterion, optimizer, train_loader):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets.float())
        loss.backward()
        optimizer.step()
    return model

# Function to evaluate the model
def evaluate_model(model, data_loader):
    model.eval()
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # Get the probabilities for the positive class (assuming outputs are logits from a binary classifier)
            probabilities = torch.sigmoid(outputs).squeeze()

            # Extend the lists to hold the true labels and the probabilities
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate the ROC-AUC score
    roc_auc = roc_auc_score(all_targets, all_probabilities)
    return roc_auc, np.array(all_targets), np.array(all_probabilities)

# Evaluation function
def evaluate(label_csv, layer, max_iter=1000, batch_size=32): # Added 'layer' as a parameter
    df = pd.read_csv(label_csv)
    df_test = df[df.is_test == True]
    df_train = df[df.is_test == False]

    model, preprocess = clip.load("ViT-B/32", device='cuda', download_root='.cache')
    train_loader = DataLoader(CLIPImageDataset(df_train["filepaths"].tolist(), preprocess), 
                            batch_size=batch_size)
    test_loader = DataLoader(CLIPImageDataset(df_test["filepaths"].tolist(), preprocess), 
                            batch_size=batch_size)
    
    train_embs = get_embs(train_loader, model, "clip")
    test_embs = get_embs(test_loader, model, "clip")

   # Get counts for each label and filter out those with fewer than 30 samples
    label_counts = df_test.iloc[:, 1:-3].apply(lambda x: x.sum(), axis=0)  # Assuming the first two columns are not labels
    valid_labels = label_counts[label_counts > 50].index.tolist()
    print(f'Found {len(valid_labels)} valid labels')
    
    # Filter the dataframe to only include valid labels
    df = df[['ImageID', 'is_test'] + valid_labels]
    target_columns = valid_labels
    target = df[target_columns].values
    is_test = df.is_test.values

    # Convert activations and labels to PyTorch tensors
    target_tensor = torch.tensor(target).float()

    # Split data into train and test
    train_activations = torch.tensor(train_embs).float()
    train_target = target_tensor[~is_test]
    test_activations = torch.tensor(test_embs).float()
    test_target = target_tensor[is_test]

    # Create Dataset and DataLoader for batching
    train_dataset = TensorDataset(train_activations, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_activations, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = train_activations.shape[1]
    output_dim = target.shape[1]  # Number of target labels

    # Initialize model, loss, and optimizer
    model = LogisticRegressionModel(input_dim, output_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Train the model
    for _ in tqdm(range(max_iter)): # Added tqdm for progress bar
        model = train(model, criterion, optimizer, train_loader)

    # Evaluate the model
    test_auc, y_true, y_pred = evaluate_model(model, test_loader)
    
    score = {
        'test_AUC': test_auc,
    }

    return model, score, y_true, y_pred

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, default='padchest', help='dataset name')
    parser.add_argument('--prompt_type', type=str, default='general', help='Type of prompt')
    parser.add_argument('--save_path', type=str, default='results/clip_alpha_0.1', help='Path where the results will be saved')
    parser.add_argument('--max_iter', type=int, default=80, help='Maximum number of iterations for training')
    parser.add_argument('--csv_path', type=str, default='csvs/padchest_.csv', help='Path to the CSV file containing labels and metadata') # Fixed here
    parser.add_argument('--layer', type=int, default=0, help='Layer of the model to evaluate') # Added layer argument
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')

    args = parser.parse_args()

    scores = []
    # Load activations

    # Train and evaluate the model
    model, score, y_true, y_pred = evaluate(
        label_csv=args.csv_path,
        layer=args.layer,  # Pass layer to evaluate function
        max_iter=args.max_iter,
        batch_size=args.batch_size  # This could be a command line argument as well
    )
    scores.append(score)

    # Save the trained probe
    save_path = save_probe(model, args.dataset, args.prompt_type, args.layer, args.save_path)
    with open(os.path.join(save_path, 'scores.json'), 'w') as f:
        json.dump(scores, f, indent=4)

    # Save the true and predicted labels
    np.save(os.path.join(save_path, 'y_true.npy'), y_true)
    np.save(os.path.join(save_path, 'y_pred.npy'), y_pred)

