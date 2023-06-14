from collections import defaultdict
from typing import List, Tuple
import pandas as pd
import json
import re

from tqdm import tqdm
from ctr_code.concatenate_highlights import combine_text_parts_to_str, concatenate_highlights_row, merge_overlapping_intervals


class Preprocessor:
    """
    Preprocess inputs and outputs
    """

    def __init__(self, prefix, special_tokens_constants, should_add_highlights: bool = True, only_sents_with_highlights: bool = False, keep_only_highlights: bool = False, add_planning_on_concatenation: bool = False, add_highlight_delim_planning: bool = False, add_highlight_labels_to_planning: bool = False, add_CoT_to_output: str = None):
        self.prefix = prefix
        self.special_tokens_constants = special_tokens_constants
        self.should_add_highlights = should_add_highlights
        self.only_sents_with_highlights = only_sents_with_highlights
        self.keep_only_highlights = keep_only_highlights
        self.add_planning_on_concatenation = add_planning_on_concatenation
        self.add_highlight_delim_planning = add_highlight_delim_planning
        self.add_highlight_labels_to_planning = add_highlight_labels_to_planning
        self.add_CoT_to_output = add_CoT_to_output

    def preprocess_input(self, source_text, highlighted_spans) -> str:
        """
        Converts input to str
        """

        # Collect all indices of tokens that need to be added
        idx_to_tokens = defaultdict(list)

        if self.keep_only_highlights:
            final_text = concatenate_highlights_row({
                "doc_text": source_text,
                "highlight_spans": highlighted_spans
            }, keep_full_sentences=False, return_str=True)
        elif self.only_sents_with_highlights:
            text_parts = concatenate_highlights_row({
                "doc_text": source_text,
                "highlight_spans": highlighted_spans
            }, keep_full_sentences=True, return_str=False)

            for text_part in text_parts:
                if text_part.is_highlight:
                    text_part.prefix = self.special_tokens_constants['highlight_start']
                    text_part.postfix = self.special_tokens_constants['highlight_end']
            final_text = combine_text_parts_to_str(text_parts, keep_full_sentences=True)
        else:
            if not self.should_add_highlights:
                highlighted_spans = []
            else:
                if isinstance(highlighted_spans, str):
                    highlighted_spans = json.loads(highlighted_spans)

                # We don't care about nested highlights / consecutive highlights
                highlighted_spans = merge_overlapping_intervals(highlighted_spans)

                for start, end in highlighted_spans:
                    idx_to_tokens[start].append(self.special_tokens_constants['highlight_start'])
                    idx_to_tokens[end].append(self.special_tokens_constants['highlight_end'])

            # Build concatenated text by running over the text in parts
            source_text_with_highlighted_spans = ""
            last_idx = 0
            for idx in sorted(idx_to_tokens.keys()):
                # Take text up to the current point
                source_text_with_highlighted_spans += source_text[last_idx:idx]

                # Add the necessary tokens
                tokens = idx_to_tokens[idx]
                for token in tokens:
                    source_text_with_highlighted_spans += token
                last_idx = idx

            source_text_with_highlighted_spans += source_text[last_idx:]

            final_text = source_text_with_highlighted_spans
            
        # Return text with prefix
        return f"{self.prefix} {final_text}"


    def preprocess_output(self, summary_text, curr_input) -> str:
        """
        Converts output to str
        """
        if self.add_planning_on_concatenation:
            all_highlights = re.findall(f"(?<={self.special_tokens_constants['highlight_start']})([\s\S]*?)(?={self.special_tokens_constants['highlight_end']})", curr_input)
            if self.add_highlight_labels_to_planning:
                all_highlights = [self.special_tokens_constants['highlight_start'] + h + self.special_tokens_constants['highlight_end'] for h in all_highlights]
            highlights_concat = self.special_tokens_constants["highlight_delim"].join(all_highlights) if self.add_highlight_delim_planning else " ".join(all_highlights)
            gold_output = self.special_tokens_constants['is_concat'] + highlights_concat + self.special_tokens_constants['is_summary'] + summary_text
        elif not self.add_CoT_to_output is None:
            if self.add_CoT_to_output == "highlights":
                all_highlights = re.findall(f"(?<={self.special_tokens_constants['highlight_start']})([\s\S]*?)(?={self.special_tokens_constants['highlight_end']})", curr_input)
                highlights_concat = "\n ".join([f"{i+1}. {h}" for i,h in enumerate(all_highlights)])
                gold_output = f"The highlighted spans are: \n{highlights_concat}\nSo, the answer is:\n {summary_text}"        
        else:
            gold_output = summary_text
        return gold_output
    

    # NEW from original script
    def preprocess_function(self, examples, tokenizer, max_source_length, max_target_length, padding="max_length", ignore_pad_token_for_loss=True, add_global_attention=False, add_global_attention_on_highlights=False, add_global_attention_on_highlighted_words=False, override_inputs=None):
        inputs, targets = [], []
        for i in range(len(examples['doc_text'])):
            # NEW from original script
            curr_input = self.preprocess_input(examples['doc_text'][i], examples['highlight_spans'][i])
            inputs.append(curr_input)
            curr_output = self.preprocess_output(examples['summary_text'][i], curr_input)
            targets.append(curr_output)

        if override_inputs is not None:
            inputs = override_inputs

        model_inputs = tokenizer(
            [x.strip() for x in inputs], max_length=max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        # NEW from original script
        global_attention_mask = []
        if add_global_attention:
            for input_ids in tqdm(model_inputs['input_ids']):
                curr_global_attention_mask = [0 for _ in range(len(input_ids))]
                curr_global_attention_mask[0] = 1

                tkns_with_global_attention = [self.special_tokens_constants[tkn_key] for tkn_key in ['highlight_start', 'highlight_end']]
                ids_with_global_attention = [special_id for special_id in tokenizer.additional_special_tokens_ids if tokenizer.convert_ids_to_tokens(special_id) in tkns_with_global_attention]
                # ids_with_global_attention = tokenizer.additional_special_tokens_ids

                highlight_start_tkn_id = tokenizer.convert_tokens_to_ids(self.special_tokens_constants['highlight_start'])
                highlight_end_tkn_id = tokenizer.convert_tokens_to_ids(self.special_tokens_constants['highlight_end'])


                if add_global_attention_on_highlights:
                    highlight_began_flag = False
                    for input_id_idx, input_id in enumerate(input_ids):
                        # Put attention on highlight tokens
                        if input_id in ids_with_global_attention: #AVIVSL: play with this (regarding the other special tokens)
                            curr_global_attention_mask[input_id_idx] = 1
                        if add_global_attention_on_highlighted_words:
                            if input_id == highlight_start_tkn_id:
                                highlight_began_flag = True
                            elif input_id == highlight_end_tkn_id:
                                highlight_began_flag = False
                            elif highlight_began_flag:
                                curr_global_attention_mask[input_id_idx] = 1

                global_attention_mask.append(curr_global_attention_mask)
            model_inputs['global_attention_mask'] = global_attention_mask

        return model_inputs

def get_special_tokens_constants(is_t5_model: bool) -> dict:
    """
    Constants used for preprocessing input and output
    """

    special_tokens_constants = {}
    if is_t5_model:
        # T5 model has 100 special tokens by default
        special_tokens_constants['highlight_start'] = "<extra_id_1>"
        special_tokens_constants['highlight_end'] = "<extra_id_2>"
        special_tokens_constants['is_concat'] = "<extra_id_3>"
        special_tokens_constants['is_summary'] = "<extra_id_4>"
        special_tokens_constants['highlight_delim'] = "<extra_id_5>"
    else:
        special_tokens_constants['highlight_start'] = "<highlight_start>"
        special_tokens_constants['highlight_end'] = "<highlight_end>"
        special_tokens_constants['is_concat'] = "<is_concat>"
        special_tokens_constants['is_summary'] = "<is_summary>"
        special_tokens_constants['highlight_delim'] = "<highlight_delim>"

    return special_tokens_constants


def convert_row_spans_str_to_list_of_highlights(spans_str) -> List[Tuple[int, int]]:
    """
    A single row's spans string can have spaces and be non-continuous. Example: "5361, 5374;5380, 5446"
    """

    highlights = []
    start_end_strs = spans_str.split(";")
    for start_end_str in start_end_strs:
        split = start_end_str.split(",")
        start = int(split[0].strip())
        end = int(split[1].strip())
        highlights.append((start, end))

    return highlights

def convert_highlight_rows_to_document_highlights(doc_reader, highlight_rows: pd.DataFrame) -> List[List[Tuple[str, str, list]]]:
    """
    Convert from multiple highlight rows (csv) to document highlights
    """

    def handle_document_rows(doc_rows):
        any_row = doc_rows.iloc[0]
        doc = doc_reader.read_doc(any_row['topic'], any_row['documentFile'])

        # Each topic is a summary
        summary = doc_reader.read_summary(any_row['summaryFile'])
        highlight_spans = doc_rows['docSpanOffsets'].apply(convert_row_spans_str_to_list_of_highlights)
        flattened_highlight_spans = [span for spans in highlight_spans.to_list() for span in spans]

        return [{
            "doc_id": any_row['documentFile'],
            "summary_id": any_row['summaryFile'],
            "doc_text": doc,
            "summary_text": summary,
            "highlight_spans": flattened_highlight_spans
        }]


    document_highlights_df = highlight_rows.groupby('summaryFile').apply(handle_document_rows)
    # Flatten list of lists to a list
    return [document_highlight for document_highlights in document_highlights_df.to_list() for document_highlight in document_highlights]


