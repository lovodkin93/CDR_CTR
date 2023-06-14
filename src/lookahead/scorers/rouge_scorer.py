import logging
import evaluate
import torch

class RougeScorer:
    """
    Scorer using Rouge
    """

    def __init__(self, device="cuda", return_negative_value=False):
        self.metric = evaluate.load('rouge')
        self.device = device
        self.return_negative_value = return_negative_value
    
    def prepare_document(self, input_str):
        """
        Prepare anything that requires processing on document.
        This is called each iteration only once to save computation.
        """
        self.input_str = input_str

    def score(self, summaries, index):
        """
        Output the score for each example.
        summaries: The summary strings
        index: The index of the example (document that it should be compared to). IT should ideally be just range() except for beam search.
        """
        logging.info(f"Rouge Scoring ; return_negative_value '{self.return_negative_value}'")

        results = self.metric.compute(predictions=summaries, references=[self.input_str[reference_idx] for reference_idx in index], use_aggregator=False)['rougeL']

        if self.return_negative_value:
            results = [-result for result in results]

        return torch.tensor(results).to(self.device)
