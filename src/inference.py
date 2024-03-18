'''
This file gets a txt file as input. The txt file has domain names.
Example script:
   python ./inference.py --input_dir /Users/marziehbitaab/Desktop/scam_propagation/Ads/saved_pages/09-18-23 --model_dir ../results/model-9-20-23 --output_file "/Users/marziehbitaab/Desktop/scam_propagation/Ads/data/shopping_ads_09-18.txt
'''
import os
os.environ['TRANSFORMERS_CACHE'] = '/mnt/marzi/hf_cache/'

import torch
import argparse
import pandas as pd
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, LongformerConfig
from datasets import Dataset
from tqdm import tqdm
from dataloader import ContentDataset
import numpy as np
import requests


# get system device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Device = %s' %(
    torch.cuda.get_device_name(device) if device != 'cpu' else 'cpu'
))
tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')


def convert_to_pd(dataset):
    # convert datasets to pandas
    data_dict = []
    for d in dataset:
        data_dict.append({
            'text': d[0],
            'label': d[1],
            'URL': d[2]
        })
    # create dataframe and HF dataset
    return pd.DataFrame(data_dict)


def tokenization(b_text):
    return tokenizer(b_text['text'], padding='max_length', truncation=True, max_length=1024)

def remove_prefix(url):
    # Lowercase the URL to handle case sensitivity
    url = url.lower()
    # Remove all known prefixes
    prefixes = ('http://', 'https://', 'www.')
    for prefix in prefixes:
        if url.startswith(prefix):
            url = url[len(prefix):]
    return url

def whoisxml(domain):

    url = f"https://website-categorization.whoisxmlapi.com/api/v3?apiKey=at_N8JkgfDnRahKpCNOjjKxVBxmCacv3&url={domain}"
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        all_cats = []
        
        # Loop through categories
        # Using .get() to avoid KeyError if 'categories' is not in response
        for category in data.get('categories', []): 
            tier1 = category.get('tier1', {})
            tier2 = category.get('tier2', {})
            
            all_cats.append(tier1.get('name', 'N/A').lower())
            
         
            if tier2:  # Check if tier2 is not None
                all_cats.append(tier2.get('name', 'N/A').lower())  

    if 'shopping' in all_cats or 'clothing' in all_cats or 'home & garden' in all_cats or 'accessories' in all_cats:
        isshop = 'xml_shop'
    else:
        isshop = 'xml_nonshop'
    return isshop,  "---".join(all_cats)

def main(args):
    # load model
    print(args.model_dir)
    model = LongformerForSequenceClassification.from_pretrained(
        './assets/model-9-20-23',
        gradient_checkpointing=False,
        attention_window=512,
    ).to(device)

    # load the dataset
    url_list = pd.read_csv(args.input_file)['URL'].tolist()
    dataset = ContentDataset({args.input_dir: -1}, url_list
    , inference_mode=True, force_build=True)
    print(len(dataset))
    pd_df = convert_to_pd(dataset)

    # convert to HF dataset and tokenize
    hf_dataset = Dataset.from_pandas(pd_df)
    test_data = hf_dataset.map(tokenization, batched=True, batch_size=1)

    # inference
    with open(args.output_file, 'w') as fout:
        # if needed add another column to output to reflec xml label
        fout.write('URL,label\n')

        for d in test_data:
            output = np.argmax(model(
                torch.tensor(d['input_ids']).unsqueeze(0).to(device), 
                torch.tensor(d['attention_mask']).unsqueeze(0).to(device)
            ).logits.tolist()[0])

            # check with whoisxml 
            #domain = remove_prefix(d['URL'])
            #whois_label, all_cats = whoisxml(domain)


            # write to file
            fout.write('%s,%s\n' %(
                d['URL'], 'shop' if output == 1 else 'nonshop'
            ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", help="directory containing HTML sources")
    parser.add_argument("--input_file", help="file containing urls")
    parser.add_argument("--model_dir", help="path to the saved model", type=str)
    parser.add_argument("--output_file", help="output file to save the results")

    args = parser.parse_args()

    main(args)
