import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import glob
from tqdm import tqdm
import logging
import json
import argparse
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import label_binarize


# Assumed device setting
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Function to load activations
def load_act(args, dataset, prompt_type, layer, base_path='activation_datasets/HuggingFaceM4/'):
    base_path = os.path.join(base_path, f"idefics-{args.param}b")
    act_paths = sorted(glob.glob(os.path.join(base_path, dataset, prompt_type, "*.pt")))
    return torch.load(act_paths[layer]).dequantize().to(device) # Make sure indexing is correct

# Function to save the probe
def save_probe(probe, dataset, prompt_type, layer, base_path):
    save_path = os.path.join(base_path, dataset, prompt_type)
    os.makedirs(save_path, exist_ok=True)
    filename = f'model_layer_{layer}.pth'
    torch.save(probe.state_dict(), os.path.join(save_path, filename))
    return save_path

# Logistic Regression Model for Multiclass
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.linear(x)  # No sigmoid, raw logits are fine here

# Function to train the model
def train(model, criterion, optimizer, train_loader):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # Use targets directly as long integers
        loss.backward()
        optimizer.step()
    return model

# Function to evaluate the model
def evaluate_model(model, data_loader, n_classes):
    model.eval()
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(probabilities, 1)

            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Calculate the accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Calculate the F1 score
    f1 = f1_score(all_targets, all_predictions, average='macro')
    
    # Binarize the labels for ROC AUC
    all_targets_binary = label_binarize(all_targets, classes=range(n_classes))
    all_probabilities = np.array(all_probabilities)

    # Calculate ROC AUC for each class
    roc_aucs = []
    if n_classes == 2:  # Special handling for binary classification
        roc_auc = roc_auc_score(all_targets, all_probabilities[:, 1])  # Use the probability of the positive class
        roc_aucs.append(roc_auc)
    else:
        for i in range(n_classes):
            # Compute ROC AUC only if there are positive instances
            if len(set(all_targets_binary[:, i])) > 1:
                roc_auc = roc_auc_score(all_targets_binary[:, i], all_probabilities[:, i])
                roc_aucs.append(roc_auc)
            else:
                roc_aucs.append(float('nan'))  # if there is no positive instance for a class, we assign NaN
    
    return accuracy, f1, roc_aucs

# Evaluation function
def evaluate(args, label_csv, acts, layer, max_iter=1000, batch_size=32):
    df = pd.read_csv(label_csv)
    
    if args.dataset == 'oai':
        df['sure'] = df['cat'] < 5
        acts = acts[df.sure.values]
        df = df[df['sure'] == True]
    
    # Assuming there is a single label column now with class indices
    target_column = 'cat'  # Change this to your actual label column name
    target = df[target_column].values
    is_test = df.is_test.values

    # Convert activations and labels to PyTorch tensors
    activations_tensor = torch.tensor(acts).float()
    target_tensor = torch.tensor(target).long()  # Labels are long integers

    # Split data into train and test
    train_activations = activations_tensor[~is_test]
    train_target = target_tensor[~is_test]
    test_activations = activations_tensor[is_test]
    test_target = target_tensor[is_test]

    # Create Dataset and DataLoader for batching
    train_dataset = TensorDataset(train_activations, train_target)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(test_activations, test_target)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = acts.shape[1]
    output_dim = len(torch.unique(target_tensor))
    print(f'Number of classes: {output_dim}')

    # Initialize model, loss, and optimizer
    model = LogisticRegressionModel(input_dim, len(torch.unique(target_tensor))).to(device)
    criterion = nn.CrossEntropyLoss()  # Changed to CrossEntropyLoss
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Train the model
    for _ in tqdm(range(max_iter)): # Added tqdm for progress bar
        model = train(model, criterion, optimizer, train_loader)

    # Evaluate the model
    test_accuracy, test_f1, test_roc_aucs = evaluate_model(model, test_loader, n_classes=output_dim)

    scores = {
        'layer': layer,
        'test_accuracy': test_accuracy,
        'test_F1': test_f1,
        'test_ROC_AUCs': test_roc_aucs,
    }

    return model, scores

# Main function
if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--prompt_type', type=str, default='general', help='Type of prompt')
    parser.add_argument('--save_path', type=str, default='results', help='Path where the results will be saved')
    parser.add_argument('--max_iter', type=int, default=80, help='Maximum number of iterations for training')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing labels and metadata') # Fixed here
    parser.add_argument('--layer', type=int, default=0, help='Layer of the model to evaluate') # Added layer argument
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for training')
    parser.add_argument('--param', type=int, default=80, help='Parameter setting')
    args = parser.parse_args()

    # Setup logging
    # logging.basicConfig(filename=os.path.join(args.save_path, 'training_log.txt'), level=logging.INFO, filemode='w')
    scores = []
    if args.param == 80:
        layer_len = 80
    elif args.param == 9:
        layer_len = 32
    for layer in range(layer_len):
        args.layer = layer
        # Load activations
        acts = load_act(args, args.dataset, args.prompt_type, args.layer)

        # Train and evaluate the model
        model, score = evaluate(
            args,
            label_csv=args.csv_path,
            acts=acts,
            layer=args.layer,  # Pass layer to evaluate function
            max_iter=args.max_iter,
            batch_size=args.batch_size  # This could be a command line argument as well
        )

        # Save the trained probe
        save_path = save_probe(model, args.dataset, args.prompt_type, args.layer, args.save_path)
        scores.append(score)
        with open(os.path.join(save_path, 'score.json'), 'w') as f:
            json.dump(scores, f, indent=4)


