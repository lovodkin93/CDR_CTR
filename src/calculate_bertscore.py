from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from bert_score import score
import argparse
import json
import torch


def main(args):
    indir = args.indir
    data_df = pd.read_csv(indir)

    with open(args.jsonl_file, 'r') as f1:
        jsonl_file_full = [json.loads(l) for l in f1.readlines()]

    bertscores = {"gold_bertscore_F1":[], "gold_bertscore_R":[], "gold_bertscore_P":[],
                  "highlights_bertscore_F1":[], "highlights_bertscore_R":[], "highlights_bertscore_P":[]}
    for index, row in tqdm(data_df.iterrows(), total=len(data_df)):
        curr_jsonl_file = jsonl_file_full[row["input_jsonl_ids"]]
        
        # Calculate BERTScore_gold
        P, R, F1 = score([row["generated_text"]], [curr_jsonl_file["gold_summary"]], lang='en', verbose=False, model_type="bert-base-uncased", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        bertscores["gold_bertscore_F1"].append(float(F1[0]))
        bertscores["gold_bertscore_R"].append(float(R[0]))
        bertscores["gold_bertscore_P"].append(float(P[0]))

        # Calculate BERTScore_highlights
        P, R, F1 = score([row["generated_text"]], [curr_jsonl_file['highlights_concatenation']], lang='en', verbose=False, model_type="bert-base-uncased", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        bertscores["highlights_bertscore_F1"].append(float(F1[0]))
        bertscores["highlights_bertscore_R"].append(float(R[0]))
        bertscores["highlights_bertscore_P"].append(float(P[0]))
    
    data_df = data_df.assign(gold_bertscore_F1=bertscores["gold_bertscore_F1"])
    data_df = data_df.assign(gold_bertscore_R=bertscores["gold_bertscore_R"])
    data_df = data_df.assign(gold_bertscore_P=bertscores["gold_bertscore_P"])
    data_df = data_df.assign(highlights_bertscore_F1=bertscores["highlights_bertscore_F1"])
    data_df = data_df.assign(highlights_bertscore_R=bertscores["highlights_bertscore_R"])
    data_df = data_df.assign(highlights_bertscore_P=bertscores["highlights_bertscore_P"])
    actual_outdir = indir.replace(".csv", "_with_bertscore.csv")
    data_df.to_csv(actual_outdir, index=False)

    # average
    bertscores_avg = {key:round(np.mean(value), 4) for key,value in bertscores.items()}
    bertscores_avg_outdir = indir.replace(os.path.basename(indir), "bertscore_results.json")
    with open(bertscores_avg_outdir, 'w') as f1:
        f1.write(json.dumps(bertscores_avg))
    print(f"Saved full results (original+bertscore for each sample) to {actual_outdir}")
    print(f"Saved averaged bertscores results to {bertscores_avg_outdir}")






if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description="")
    argparser.add_argument('-i', '--indir', type=str, default=None, help='path to csv with geneerated outputs - to which a "bertscore" column will be added.')
    argparser.add_argument('--jsonl-file', type=str, default=None, help='path to the jsonl file (where the "gold outputs" and the "highlight concatentations" are)')

    args = argparser.parse_args()
    main(args)














