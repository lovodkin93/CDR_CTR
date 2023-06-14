import logging
import evaluate
import torch

class MeteorScorer:
    """
    Scorer using Meteor
    """

    def __init__(self, device="cuda"):
        self.metric = evaluate.load('meteor')
        self.device = device
    
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
        logging.info("Rouge Scoring")

        references = [self.input_str[reference_idx] for reference_idx in index]
        assert len(summaries) == len(references)
        results = [self.metric.compute(predictions=[summary], references=[reference])['meteor'] for summary, reference in zip(summaries, references)]
        return torch.tensor(results).to(self.device)
