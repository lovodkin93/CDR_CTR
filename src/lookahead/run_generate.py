import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from lookahead import Lookahead
from generation import Generator
from tqdm import tqdm
import pandas as pd
from ctr_code.concatenate_highlights import concatenate_highlights
import torch
from accelerate import Accelerator

import json

import argparse

from ctr_code.preprocessor import Preprocessor, get_special_tokens_constants
from ctr_code.QUARK.utils.utils import add_control_code
from ctr_code.QUARK.policy import Policy
from consts import model_name_to_path
from scorers.sum_scorer import SumScorer
from scorers.iterative_scorer import IterativeScorer
from scorers.bert_score_scorer import BERTScoreScorer
from scorers.rouge_scorer import RougeScorer
from scorers.meteor_scorer import MeteorScorer

BEST_CAT_ID = ' _TREE_TOKEN_00000'

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format= '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
        
    )
    
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()

    # base decoding model
    parser.add_argument("--model_name", type=str, default="flan")
    parser.add_argument("--cache_dir", type=str, default="./cache")

    # input output
    parser.add_argument("--document_file", type=str, required=False)
    parser.add_argument("--hf_dataset", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--compare_highlights", type=bool, default=True)
    parser.add_argument("--split", type=str, default="eval")

    # base decoding configuration. Please refer to Huggingface's GenerationMixin for the explaination of the parameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--max_examples", type=int, default=1000)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    parser.add_argument("--max_input_length", type=int, default=1400)
    parser.add_argument("--max_output_length", type=int, default=512)
    parser.add_argument("--min_length", type=int, default=100)
    parser.add_argument("--length_penalty", type=int, default=2.0)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--do_sample", action='store_true', default=False)

    # lookahead configuration
    parser.add_argument("--do_lookahead", action="store_true", default=False)
    parser.add_argument("--lookahead_length", type=int, default=64)
    parser.add_argument("--lookahead_lambda", type=float, default=25)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--lookahead_decoding_type", type=str, default="greedy", choices=["greedy","beam","sample"])
    parser.add_argument("--lookahead_beam", type=int, default=1)

    # scorer configuration
    parser.add_argument("--scorer", type=str, default="bert_score")
    parser.add_argument("--scorer_model_type", type=str, default="roberta-large")  
    parser.add_argument("--scorer_num_layers", type=int, default=17)

    args = parser.parse_args()

    # loading model
    model_name_or_path = model_name_to_path[args.model_name] if args.model_name in model_name_to_path else args.model_name
    print(f"Loading model {model_name_or_path}")

    accelerator = Accelerator(cpu=False)

    is_quark_model = args.model_name.endswith('quark')
    if is_quark_model:
        quark_model_name_or_path = model_name_or_path
        model_name_or_path = model_name_to_path[f"{args.model_name}_init"]
        # model_name_or_path = model_name_to_path[args.model_name.split("_quark")[0]]

        tree_tokens = [' _TREE_TOKEN_{}'.format(str(idx).zfill(5)) for idx in range(8)] + \
                [' _TREE_TOKEN_ZERO_COMMENTS'] # tokens of the "categories" --> i.e., the quantization quantiles
        
        with open(f"{quark_model_name_or_path}/args.json") as f:
            quark_args = json.load(f)
            quark_args = argparse.Namespace(**quark_args)

        Policy(model_name=model_name_or_path, temperature=quark_args.temperature, device='cuda', args=quark_args, logger=logger, last_checkpoint=None, accelerator=accelerator)
        policy = Policy(model_name=model_name_or_path, temperature=quark_args.temperature, device='cuda',
                        reward_cond=True, tree_tokens=tree_tokens, args=quark_args, logger=logger, last_checkpoint=None, accelerator=accelerator)
        model = policy.model
        tokenizer = policy.tokenizer

    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir=args.cache_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, cache_dir=args.cache_dir)
        model = model.cuda() # can optionally call .half() for mixed precision

    if is_quark_model:
        accelerator.load_state(quark_model_name_or_path)

    # loading input
    if args.hf_dataset:
        import datasets
        # dataset = datasets.load_dataset(args.hf_dataset, name="CNN-DM")
        # documents = [document.strip() for document in dataset['test']['document']]
        file = "dev__highlights.csv" if args.split == "eval" else "test__highlights.csv"
        dataset_df = pd.read_csv(f"dataset/{file}")

        documents = dataset_df

        if args.max_examples is not None:
            documents = documents[:args.max_examples]

        # Inputs are different
        if is_quark_model:
            base_model_name_or_path = args.model_name.replace("_quark", "").replace("_distilled", "")
            file = f"dataset/{args.split}_set_{base_model_name_or_path}.jsonl"
            with open(file, "r") as f:
                items = [json.loads(line) for line in f.readlines()]

            quark_inputs = [item['input'] for item in items]

        # We need to take the unhighlights from the quark jsonl
        if args.scorer in ["neg_rouge", "iterative_rouge"]:
            file = f"dataset/{args.split}_set_led.jsonl"
            with open(file, "r") as f:
                items = [json.loads(line) for line in f.readlines()]

            documents['unhighlights_concatenation'] = [item['unhighlights_concatenation'] for item in items]

    else:
        raise ValueError()

    #  Create lookahead
    lookahead = None
    if args.do_lookahead:
        if args.scorer == "bert_score":
            # Load scorer for lookahead
            scorer = BERTScoreScorer(
                model_name=args.scorer_model_type,
                num_layers=args.scorer_num_layers,
                cache_dir=args.cache_dir,
                # device="cuda:1"  # Run on a different gpu
            )
        elif args.scorer == "meteor":
            # Load scorer for lookahead
            scorer = MeteorScorer()
        elif args.scorer == "neg_rouge":
            scorer = RougeScorer(return_negative_value=True)
        elif args.scorer == "iterative_rouge":
            scorers = [RougeScorer(), RougeScorer(return_negative_value=True)]
            scorer = IterativeScorer(scorers)
        elif args.scorer == "sum_rouge_meteor":
            scorers = [RougeScorer(), MeteorScorer()]
            scorer = SumScorer(scorers)
        else:
            scorer = RougeScorer()

        lookahead = Lookahead(
            model,
            tokenizer,
            scorer,
            lookahead_length=args.lookahead_length,
            lookahead_lambda=args.lookahead_lambda,
            lookahead_top_k=args.top_k,
            decoding_type=args.lookahead_decoding_type,
            num_beams=args.lookahead_beam,
            num_return_sequences=args.lookahead_beam,
            max_length=args.max_output_length,
            min_length=args.min_length,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size
        )

    # Create generator with lookahead
    generator = Generator(model, lookahead=lookahead)

    if is_quark_model:
        policy.model = generator

    summaries = []

    is_t5_model = args.model_name in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ] or model.config.model_type == 't5'
    special_tokens_constants = get_special_tokens_constants(is_t5_model)
    prefix = ""
    if is_t5_model:
        prefix = "Instruction: In this task, you are presented with a passage, where some parts are \"highlighted\" (namely, there are <extra_id_1> and <extra_id_2> tokens before and after each such span). \nYour job is to generate a summary that covers all and only the \"highlighted\" spans. \nPassage: "
    preprocessor = Preprocessor(prefix, special_tokens_constants)

    if args.do_lookahead:
        params_str = f"{args.split}__{args.model_name}__num_beams_{args.num_beams}__scorer_{args.scorer}__do_lookahead"
    else:
        params_str = f"{args.split}__{args.model_name}__num_beams_{args.num_beams}__no_lookahead"
    output_file = f"{args.output_dir}/{params_str}.json"

    for i in tqdm(range(0, len(documents), args.batch_size)):
        input_str = documents[i:i+args.batch_size]
        
        # Necessary for methods enumerating the dataframe length
        input_str = input_str.reset_index()
        
        override_inputs = None
        if is_quark_model:
            override_inputs = quark_inputs[i:i+args.batch_size]

        is_led_model = model.config.model_type == 'led'
        add_global_attention = False
        add_global_attention_on_highlights = False
        if is_led_model:
            add_global_attention = True
            add_global_attention_on_highlights = True

        padding = "max_length"
        if is_quark_model:
            padding = True

        inputs = preprocessor.preprocess_function(input_str, tokenizer, args.max_input_length, args.max_output_length, add_global_attention=add_global_attention, add_global_attention_on_highlights=add_global_attention_on_highlights, override_inputs=override_inputs, padding=padding)
        inputs['input_ids'], inputs['attention_mask'] = torch.tensor(inputs['input_ids']), torch.tensor(inputs['attention_mask'])

        if is_led_model:
            inputs['global_attention_mask'] = torch.tensor(inputs["global_attention_mask"])

        if is_quark_model:
            best_cat_id_encoded = tokenizer.convert_tokens_to_ids(BEST_CAT_ID)
            if is_led_model:
                inputs['input_ids'], inputs['attention_mask'], inputs['global_attention_mask'] = add_control_code(inputs['input_ids'], inputs['attention_mask'], best_cat_id_encoded, inputs['global_attention_mask'])
            else:
                inputs['input_ids'], inputs['attention_mask'], _ = add_control_code(inputs['input_ids'], inputs['attention_mask'], best_cat_id_encoded)

        # For efficency, already prepare the references that the predictions will be compared to (e.g., create BERT embeddings)
        if generator.lookahead is not None:
            reference_str = input_str
            if args.compare_highlights:
                if args.scorer == "neg_rouge":
                    reference_str = input_str['unhighlights_concatenation'].tolist()
                elif args.scorer in ["iterative_rouge"]:
                    highlights_reference = concatenate_highlights(input_str)
                    neg_reference = input_str['unhighlights_concatenation'].tolist()
                    reference_str = [highlights_reference, neg_reference]
                elif args.scorer in ["sum_rouge_meteor"]:
                    highlights_reference = concatenate_highlights(input_str)
                    reference_str = [highlights_reference, highlights_reference]
                else:
                    reference_str = concatenate_highlights(input_str)

            generator.lookahead.scorer.prepare_document(reference_str)

        extra_args = {}
        if is_led_model:
            extra_args['global_attention_mask'] = inputs["global_attention_mask"].cuda()

        if is_quark_model:
            rollouts = policy.sample(input_ids=inputs["input_ids"].cuda(),
                                     attention_mask=inputs["attention_mask"].cuda(),
                                     sample=args.do_sample,
                                     top_p=quark_args.top_p,
                                     **extra_args)
            output = rollouts['output_reduction/text']
        else:
            output = generator.generate(
                input_ids=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                max_length=args.max_output_length,
                do_sample=args.do_sample,
                **extra_args
            )

            output = tokenizer.batch_decode(output, skip_special_tokens=True)
        
        if args.num_return_sequences == 1:
            summaries += output
        else:
            for i in range(0, len(output), args.num_return_sequences):
                summaries.append(output[i:i+args.num_return_sequences])

        # Save file
        with open(output_file, "w") as f:
            if args.num_return_sequences == 1:
                for line in summaries:
                    f.write(line.replace("\n", "<newline>") + "\n")
            else:
                json.dump(summaries, f)