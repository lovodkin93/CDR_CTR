import torch
from tqdm import tqdm

class PromptCollator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.max_source_length = args.max_source_length if hasattr(args, 'max_source_length') else None
        self.add_global_attention = args.add_global_attention
        self.add_global_attention_on_highlights = args.add_global_attention_on_highlights
        self.input_prefix = f"{args.source_prefix.strip()} " if args.source_prefix is not None else "" # relevant for t5 models

    def __call__(self, sequences):
        input_text = [self.input_prefix + sequence['input'] for sequence in sequences]
        jsonl_ids = [sequence['jsonl_id'] for sequence in sequences] # index of input texts in the input dataset

        encodings_dict = self.tokenizer(input_text, max_length= self.max_source_length, padding=True, truncation=True) 
        input_ids = torch.as_tensor(encodings_dict['input_ids'])
        attention_mask = torch.as_tensor(encodings_dict['attention_mask'])
        
        global_attention_mask = None
        if self.add_global_attention:
            global_attention_mask = []
            for input_ids_instance in tqdm(encodings_dict['input_ids']):
                curr_global_attention_mask = [0 for _ in range(len(input_ids_instance))]
                curr_global_attention_mask[0] = 1

                ids_with_global_attention = self.tokenizer.additional_special_tokens_ids

                if self.add_global_attention_on_highlights:
                    for input_id_idx, input_id in enumerate(input_ids_instance):
                        # Put attention on highlight tokens
                        if input_id in ids_with_global_attention: 
                            curr_global_attention_mask[input_id_idx] = 1

                global_attention_mask.append(curr_global_attention_mask)
            global_attention_mask = torch.as_tensor(global_attention_mask)
        return input_ids, attention_mask, global_attention_mask, jsonl_ids
