import numpy as np

from src.utils import filter_function_words
from bert_score import score as bertscore
import torch

def compute_rouge_metrics(predictions: list, references: list, metric, prefix: str, should_filter_function_words: bool = False) -> dict:
    assert len(predictions) == len(references)

    filtered_predictions = predictions
    filtered_references = references
    if should_filter_function_words:
        filtered_predictions = []
        for prediction in predictions:
            filtered_predictions.append(filter_function_words(prediction))

        filtered_references = []
        for reference in references:
            filtered_references.append(filter_function_words(reference))
    result = metric.compute(predictions=filtered_predictions,
                            references=filtered_references, use_stemmer=True)
    # Extract a few results from ROUGE
    result_parsed = {f"{prefix}_{key}": value*100 for key, value in result.items()}

    # Add also precision and recall
    # result_parsed.update({f"{prefix}_{key}_precision": value.mid.precision * 100 for key, value in result.items()})
    # result_parsed.update({f"{prefix}_{key}_recall": value.mid.recall * 100 for key, value in result.items()})

    result_parsed = {k: round(v, 4) for k, v in result_parsed.items()}

    return result_parsed

def compute_meteor_metrics(predictions: list, references: list, metric, prefix: str) -> dict:
    result = metric.compute(predictions=predictions, references=references)
    return {
        f"{prefix}meteor": round(result['meteor']*100, 4)
    }


def compute_bertscore_metrics(predictions: list, references: list, prefix: str) -> dict:
        # Calculate BERTScore_gold
        P, R, F1 = bertscore(predictions, references, lang='en', verbose=False, model_type="bert-base-uncased", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        return {
        f"{prefix}bertscore_F1": round(np.mean([float(scr)*100 for scr in F1]), 4),
        f"{prefix}bertscore_P": round(np.mean([float(scr)*100 for scr in P]), 4),
        f"{prefix}bertscore_R": round(np.mean([float(scr)*100 for scr in R]), 4),
        }

        # return {
        # f"{prefix}F1": round(result['meteor']*100, 4)
        # }
        
        # bertscores["gold_bertscore_F1"].append(float(F1[0]))
        # bertscores["gold_bertscore_R"].append(float(R[0]))
        # bertscores["gold_bertscore_P"].append(float(P[0]))
