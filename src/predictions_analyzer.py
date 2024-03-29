import os
import sys
from datasets import load_metric
import pandas as pd
import numpy as np    
import json
import re
from src.concatenate_highlights import concatenate_highlights


class PredictionsAnalyzer:
    """
    Extracts an analyzed result for each prediction instead of an aggregate of all predictions
    """

    def __init__(self, tokenizer, preprocessor, is_add_planning_on_concatenation, output_dir: str, rouge_metric) -> None:
        self.tokenizer = tokenizer
        self.preprocessor = preprocessor
        self.is_add_planning_on_concatenation = is_add_planning_on_concatenation
        self.output_dir = output_dir
        self.rouge_metric = rouge_metric

    def write_predictions_to_file(self, predictions, dataset, df, is_tokenized=True):
        objects = self._clean_predictions(predictions, dataset, is_tokenized)

        # Calculate rouge between gold and summaries (if there is gold)
        if objects.get('gold') is not None:
            self.calculate_rouge_between_gold_n_prediction(objects, objects['predicted'], objects['gold'], prefix="gold")

        # Calculate rouge between input and summary
        highlights_input = concatenate_highlights(df)
        self.calculate_rouge_between_gold_n_prediction(objects, objects['predicted'], highlights_input, prefix="highlights")
        self._save_to_file(objects)

    def calculate_rouge_between_gold_n_prediction(self, objects, decoded_predictions, gold, prefix: str):
        result_per_pred = self.rouge_metric.compute(predictions=decoded_predictions, references=gold, use_stemmer=True, use_aggregator=False)
        objects[f'{prefix}_rouge1'] = [x for x in result_per_pred['rouge1']]
        objects[f'{prefix}_rouge2'] = [x for x in result_per_pred['rouge2']]
        objects[f'{prefix}_rougeL'] = [x for x in result_per_pred['rougeL']]

    def _clean_predictions(self, predictions, dataset, is_tokenized):
        
        def remove_special_tokens(curr_preds):
            all_special_tkns = sum([special_tkns if type(special_tkns)==list else [special_tkns] for special_tkns in self.tokenizer.special_tokens_map.values()], [])
            start_summary_tkn = self.preprocessor.special_tokens_constants["is_summary"]
            curr_preds = [re.sub(r'|'.join(map(re.escape, all_special_tkns)), '', pred) for pred in curr_preds] # remove the special tokens
            return curr_preds


        def remove_pad_tokens(prediction_tokens):
            """
            We want to calculate the num of tokens without the padding
            """

            return [token for token in prediction_tokens if token != self.tokenizer.pad_token_id]

        # Non-tokenized can be outputs not from a model, such as naive concatenation
        if not is_tokenized:
            decoded_predictions = predictions
            input_seqs = None
            clean_input_seqs = dataset
            input_tokenizer_lengths = None
            predictions_tokenizer_lengths = None
        else:
            decoded_predictions = self.tokenizer.batch_decode(predictions)
            
            if self.is_add_planning_on_concatenation:
                decoded_predictions_two_parts = [pred.split(self.preprocessor.special_tokens_constants["is_summary"]) for pred in decoded_predictions]
                decoded_predictions = [elem[-1] for elem in decoded_predictions_two_parts] # take only the part of the summary, without the concatenation
                predicted_concat = [elem[0] if len(elem)==2 else None for elem in decoded_predictions_two_parts] # if doesn't learn to begin with the concatenation - then the "separation" into two parts won't yield two parts but rather one
            
            decoded_predictions = remove_special_tokens(decoded_predictions)
            decoded_predictions = [pred.strip() for pred in decoded_predictions]

            input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'])
                            for i in range(len(dataset))]
            clean_input_seqs = [self.tokenizer.decode(dataset[i]['input_ids'], skip_special_tokens=True)
                            for i in range(len(dataset))]

            # Length can be useful to see if the model actually saw everything
            predictions_tokenizer_lengths = [len(remove_pad_tokens(predictions[i])) for i in range(len(predictions))]
            input_tokenizer_lengths = [len(dataset[i]['input_ids']) for i in range(len(dataset))]

        gold = None
        gold_tokenizer_lengths = None
        if 'labels' in dataset[0]:
            gold = [self.tokenizer.decode(dataset[i]['labels']) for i in range(len(dataset))]
            if self.is_add_planning_on_concatenation:
                gold = [pred.split(self.preprocessor.special_tokens_constants["is_summary"])[-1] for pred in gold] # take only the part of the summary, without the concatenation
            gold = remove_special_tokens(gold)
            # Length can be useful to see if the model actually saw everything
            gold_tokenizer_lengths = [len(dataset[i]['labels']) for i in range(len(dataset))]

        objects = {"input": input_seqs, "clean_input": clean_input_seqs, "input_tokenizer_length": input_tokenizer_lengths, "predicted": decoded_predictions, "prediction_tokenizer_length": predictions_tokenizer_lengths}
        if self.is_add_planning_on_concatenation:
            objects["predicted_concat_part"] = predicted_concat
        if gold is not None:
            objects["gold"] = gold
            objects["gold_tokenizer_length"] = gold_tokenizer_lengths

        return objects

    def _save_to_file(self, objects):
        df = pd.DataFrame(objects)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        output_prediction_file = os.path.join(
            self.output_dir, "generated_predictions.csv")
        df.to_csv(output_prediction_file, index=False)
