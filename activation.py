import os
import argparse
from tqdm import tqdm
from PIL import Image
import pandas as pd

import torch
import einops
import datasets
from torch.utils.data import Dataset
from transformers import IdeficsForVisionText2Text, AutoProcessor, BitsAndBytesConfig, AutoTokenizer
from utils import adjust_precision

PROMPT = {
    'patho_kather': 'Question: What can you seen on this histological picture? Answer:',
    'pannuke': 'Question: What can you seen on this histological picture? Answer:',
    'oai': 'Question: What can you seen on this knee radiological image? Answer:',
    'padchest': 'Question: What can you seen on this chest radiological image? Answer:',
    'isic': 'Question: What can you seen on this skin image? Answer:',
    'eye': 'Question: What can you seen on this eye fundus image? Answer:',
    'general': 'Question: What\'s on this image? Answer:',
}


class patchdataset(Dataset):
    def __init__(self, df_path, split=None, prompt='', sample=0):
        self.df = pd.read_csv(df_path)
        self.prompt = prompt
        if split is not None:
            self.df = self.df[self.df['split'] == split]
        if sample > 0:
            self.df = self.df.sample(sample)
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_path = row['filepaths']
        image = Image.open(image_path)
        return {'image': image, 'prompt': self.prompt}


def process_activation_batch(args, batch_activations, step, batch_mask=None):
    # TODO: understand batch_mask
    cur_batch_size = batch_activations.shape[0]

    if args.activation_aggregation is None:
        # only save the activations for the required indices
        batch_activations = einops.rearrange(
            batch_activations, 'b c d -> (b c) d')  # batch, context, dim
        processed_activations = batch_activations[-1, :]
        processed_activations = processed_activations.unsqueeze(0)

    if args.activation_aggregation == 'last':
        print('before rearrange', batch_activations.shape)
        last_ix = batch_activations.shape[1] - 1
        batch_mask = batch_mask.to(int)
        last_entity_token = last_ix - \
            torch.argmax(batch_mask.flip(dims=[1]), dim=1)
        d_act = batch_activations.shape[2]
        expanded_mask = last_entity_token.unsqueeze(-1).expand(-1, d_act)
        processed_activations = batch_activations[
            torch.arange(cur_batch_size).unsqueeze(-1),
            expanded_mask,
            torch.arange(d_act)
        ]
        assert processed_activations.shape == (cur_batch_size, d_act)

    return processed_activations

def make_token_dataset(text, image_paths, processor, tokenizer):
    # TODO: check entity_mask
    ids, masks, pixels = [], [], []
    # masks_a, masks_i = [], []
    
    for image_path in tqdm(image_paths, desc='Tokenizing images'): 
        prompt = [
            Image.open(image_path),
            # "Question: What's on the picture? Answer:",
            text,
        ]

        inputs = processor(prompt, return_tensors="pt")
        token_ids = inputs.input_ids

        # add bos token
        token_ids = torch.cat([
            torch.ones(token_ids.shape[0], 1,
                    dtype=torch.long) * tokenizer.bos_token_id,
            token_ids], dim=1
        )

        prompt_tokens = (token_ids[0] == token_ids).all(axis=0)
        entity_mask = torch.ones_like(token_ids, dtype=torch.bool)
        entity_mask[:, prompt_tokens] = False
        entity_mask[token_ids == tokenizer.pad_token_id] = False

        ids.append(token_ids)
        masks.append(entity_mask)
    
    dataset = datasets.Dataset.from_dict({
        'input_ids': ids,
        'entity_mask': masks,
    })

    dataset.set_format(type='torch', columns=['entity_mask', 'input_ids'])
    return dataset


@torch.no_grad()
def get_layer_activations_hf(
    args, model, tokenized_dataset, processor, layers='all', device=None,
):
    if layers == 'all':
        layers = list(range(model.config.num_hidden_layers))
    if device is None:
        device = model.device

    entity_mask = torch.tensor(tokenized_dataset['entity_mask'])

    n_seq, _, ctx_len = tokenized_dataset['input_ids'].shape
    activation_rows = n_seq
    print(f'activation_rows: {activation_rows}')
    layer_activations = {
        l: torch.zeros(activation_rows, model.config.hidden_size, # e.g. (100, 8192)
                       dtype=torch.float16)
        for l in layers
    }
    offset = 0
    bs = 1
    patch_dataset = patchdataset(args.csv_file,
                                 prompt=args.prompt,
                                 )

    for step, item in enumerate(tqdm(patch_dataset, disable=False)):
        txt = item['prompt']
        image = item['image']

        # clip batch to remove excess padding
        batch_entity_mask = entity_mask[step*bs:(step+1)*bs]
        last_valid_ix = torch.argmax(
            (batch_entity_mask.sum(dim=0) > 0) * torch.arange(ctx_len)) + 1
        batch_entity_mask = batch_entity_mask[:, :last_valid_ix]

        prompt = [
            image,
            txt,
        ]
        input = processor(prompt, return_tensors="pt") 

        out = model(**input,
            output_hidden_states=True,
            output_attentions=False, 
            return_dict=True, 
            use_cache=False
        )
        # do not save post embedding layer activations
        for lix, activation in enumerate(out.hidden_states[1:]):
            if lix not in layer_activations:
                continue
            activation = activation.cpu().to(torch.float16)
            processed_activations = process_activation_batch(
                args, activation, step, batch_entity_mask
            )

            # TODO: not understanding this
            save_rows = processed_activations.shape[0]
            layer_activations[lix][offset:offset + save_rows] = processed_activations

        offset += bs

    return layer_activations


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment params
    parser.add_argument(
        '--model', default='HuggingFaceM4/idefics-80b',
        help='Name of model from TransformerLens')
    parser.add_argument(
        '--entity_type',
        help='Name of entity_type')
    parser.add_argument(
        '--activation_aggregation', default=None,
        help='Average activations across all tokens in a sequence')
    # base experiment params
    parser.add_argument(
        '--device', default="cuda" if torch.cuda.is_available() else "cpu",
        help='device to use for computation')
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='batch size to use for model.forward')
    parser.add_argument(
        '--save_precision', type=int, default=8, choices=[8, 16, 32],
        help='Number of bits to use for saving activations')
    parser.add_argument(
        '--n_threads', type=int,
        default=int(os.getenv('SLURM_CPUS_PER_TASK', 8)),
        help='number of threads to use for pytorch cpu parallelization')
    parser.add_argument(
        '--layers', nargs='+', type=int, default=None)
    parser.add_argument(
        '--use_tl', action='store_true',
        help='Use TransformerLens model instead of HuggingFace model')
    parser.add_argument(
        '--is_test', action='store_true')
    parser.add_argument(
        '--prompt_name', default='all')
    parser.add_argument(
        '--csv_file', default='./csvs/patho_kather.csv')
    parser.add_argument(
        '--prompt_key', default='general')
    parser.add_argument(
        '--prompt', default=None)

    args = parser.parse_args()
    args.prompt = PROMPT[args.prompt_key]

    print('Begin loading model')
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    model = IdeficsForVisionText2Text.from_pretrained(args.model, 
                                                      quantization_config=quantization_config, 
                                                      device_map="auto")
    processor = AutoProcessor.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print('Finished loading model')

    torch.set_grad_enabled(False)
    model.eval() # check if this is necessary

    df = pd.read_csv(args.csv_file)
    tokenized_dataset = make_token_dataset(text=args.prompt,
                                             image_paths=df['filepaths'].tolist(), 
                                             processor=processor, 
                                             tokenizer=tokenizer)
    
    activation_save_path = os.path.join(
        os.getenv('ACTIVATION_DATASET_DIR', 'activation_datasets'),
        args.model,
        args.csv_file.split('/')[-1].split('.')[0],
        args.prompt_key,
    )
    os.makedirs(activation_save_path, exist_ok=True)

    print(f'Begin processing {args.model} {args.prompt_key}')

    layer_activations = get_layer_activations_hf(
        args, model, tokenized_dataset, processor,
        device=args.device,
    )

    for layer_ix, activations in layer_activations.items():
        save_name = f'{args.entity_type}.{layer_ix}.pt'
        save_path = os.path.join(activation_save_path, save_name)
        activations = adjust_precision(
            activations.to(torch.float32), args.save_precision, per_channel=True)
        torch.save(activations, save_path)
