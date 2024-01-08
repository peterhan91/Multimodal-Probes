import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import clip
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import argparse

from sklearn.linear_model import SGDClassifier
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score


######## run clip baseline classiifier ########
# output of this script should contain 
# 1. performance.json
# 2. prediction prob and label json


def eval_metrics(y_true, y_pred, average_method=None):
    assert len(y_true) == len(y_pred)
    f1 = np.mean(f1_score(y_true, y_pred, average = average_method))
    performance = {'F1': f1,
                   'instances' : len(y_true)}
    return performance

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


def run_classification(args, train_x, train_y, test_x, test_y, seed=1, alpha=0.1):
    classifier = SGDClassifier(random_state=seed, loss="log_loss",
                               alpha=alpha, verbose=0,
                               penalty="l2", max_iter=10000, class_weight="balanced")
    
    le = LabelEncoder()

    if args.subsample < 1:
        np.random.seed(1)
        idx = np.random.choice(train_x.shape[0], int(args.subsample*train_x.shape[0]), replace=False)
        train_x = train_x[idx]
        train_y = np.array(train_y)[idx].tolist()

    train_y = le.fit_transform(train_y)
    test_y = le.transform(test_y)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    classifier.fit(train_x, train_y)
    test_pred = classifier.predict(test_x)
    test_pred_p = classifier.predict_proba(test_x)
    test_metrics = eval_metrics(test_y, test_pred)

    return test_metrics, test_y, test_pred, test_pred_p
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--csv_path', type=str, required=True, help='Path to the CSV file containing labels and metadata') # Fixed here
    parser.add_argument('--dataset', type=str, required=True, help='dataset name')
    parser.add_argument('--save_path', type=str, default='results_subsample', help='Path where the results will be saved')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--subsample', type=float, default=1, required=True, help='Subsample activations')
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if args.dataset == 'oai':
        df['sure'] = df['cat'] < 5
        df = df[df['sure'] == True]
    elif args.dataset == 'diabetic':
        df['airogs'] = df['dataset'] == 'airogs'
        df = df[df['airogs'] == False]  # Filter out airogs data
        df['cat'] = df['score']
    elif args.dataset == 'glaucoma':
        df['airogs'] = df['dataset'] == 'airogs'
        df_airogs = df[df['airogs'] == True]  # airogs data
        eyes = {
        6:'no referable glaucoma',
        7:'referable glaucoma',
        }
        df_airogs['class'] = df_airogs['score'].map(eyes)
        df_airogs['cat'] = df_airogs['class'].apply(lambda x: 1 if x=='referable glaucoma' else 0)
        df_airogs['is_test'] = False
        df_airogs = df_airogs[['filepaths', 'class', 'cat', 'is_test']]
        df_airogs['dataset'] = 'airogs'
        df_odir = pd.read_csv('csvs/odir.csv') # filepaths, class, cat, is_test, dataset
        df = pd.concat([df_airogs, df_odir], axis=0)

    train_dataset = df[df['is_test'] == False]
    test_dataset = df[df['is_test'] == True]

    test_y = test_dataset["cat"].tolist()
    train_y = train_dataset["cat"].tolist()

    def run_study(model_name, cache_dir=".cache", dataset=None):
        if model_name == "clip":
            model, preprocess = clip.load("ViT-B/32", device='cuda', download_root=cache_dir)
        else:
            ValueError("Model not supported")
        
        train_loader = DataLoader(CLIPImageDataset(train_dataset["filepaths"].tolist(), preprocess), 
                                  batch_size=args.batch_size, num_workers=32)
        test_loader = DataLoader(CLIPImageDataset(test_dataset["filepaths"].tolist(), preprocess), 
                                 batch_size=args.batch_size, num_workers=32)

        train_embs = get_embs(train_loader, model, model_name)
        test_embs = get_embs(test_loader, model, model_name)
        
        output = {}
        alpha = 0.001
        metrics, y_true, y_pred, y_pred_prob = run_classification(args, train_embs, 
                                                                  train_y, test_embs, test_y, alpha=alpha)
        output['y_true'] = y_true.tolist()
        output['y_pred'] = y_pred.tolist()
        output['y_pred_prob'] = y_pred_prob.tolist()

        metrics["alpha"] = alpha
        metrics["model_name"] = model_name
        save_path = os.path.join(args.save_path, model_name, 
                                 dataset, 'general', str(args.subsample))
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, "performance.json"), "w") as f:
            json.dump(metrics, f, indent=4)

        with open(os.path.join(save_path, "prediction.json"), "w") as f:
            json.dump(output, f, indent=4)

    run_study("clip", cache_dir=".cache", dataset=args.dataset)


