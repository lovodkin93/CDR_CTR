import logging
import evaluate
import torch

from scorers.rouge_scorer import RougeScorer

class SumScorer:
    """
    Sum multiple scorers
    """

    def __init__(self, scorers):
        self.scorers = scorers
        self.scorer_idx = 0
    
    def prepare_document(self, input_str):
        """
        Prepare anything that requires processing on document.
        This is called each iteration only once to save computation.
        """
        
        for scorer_idx, scorer in enumerate(self.scorers):
            scorer.prepare_document(input_str[scorer_idx])

    def score(self, summaries, index):
        """
        Iteratively switch scorers
        """
        logging.info("Sum scoring")

        return sum(scorer_to_use.score(summaries, index) for scorer_to_use in self.scorers)
