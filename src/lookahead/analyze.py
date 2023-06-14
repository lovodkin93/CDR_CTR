import json
import pandas as pd
import evaluate
from lookahead.consts import model_name_to_path
from lookahead.ctr_code.concatenate_highlights import concatenate_highlights


def read_file(output_file):
    with open(output_file, "r") as f:
        lines = [lines for lines in f.readlines()]
    
    return lines

def read_highlights_concatenation(existing_preds, split):
    # Read file
    file = "dev__highlights.csv" if split == "eval" else "test__highlights.csv"
    dataset_df = pd.read_csv(f"dataset/{file}")

    # Add concatenated highlights to existing_preds
    existing_preds['highlights_concatenated'] = concatenate_highlights(dataset_df)

    return existing_preds

def fix_newlines(preds):
    """
    when saving to file we added <newline> instead of \n to be able to read the file properly.
    Here we change it to spaces for the metrics calculation
    """

    return [x.replace("<newline>", " ") for x in preds]


if __name__ == "__main__":
    split = "eval"
    # split = "test"

    # model_name = "led_distilled_quark"
    model_name = "flan"
    model_path = model_name_to_path[model_name]

    # Num beams 2 is to validate output, num beams 8 is the actual output (one with lookahead, one without)
    # output_file = f"output/{split}__{model_name}__num_beams_2__no_lookahead.json"
    # output_file = f"output/{split}__{model_name}__num_beams_8__no_lookahead.json"
    output_file = f"output/{split}__{model_name}__num_beams_8__scorer_sum_rouge_meteor__do_lookahead.json"

    is_quark_model = "quark" in output_file

    is_the_same_model = True
    try:
        existing_preds = pd.read_csv(f"{model_path}/generated_predictions_{split}.csv")
    except:
        print("not comparing to baseline predictions")
        is_the_same_model = False
        base_model_name = model_name.replace("_quark", "").replace("_distilled", "")
        other_model_path = model_name_to_path[base_model_name]
        existing_preds = pd.read_csv(f"{other_model_path}/generated_predictions_{split}.csv")

        # If not the same model, use the file just for the input and gold, not for the output
        COLUMNS_TO_KEEP = ['input', 'clean_input', 'gold']
        existing_preds = existing_preds[COLUMNS_TO_KEEP]

    # Quark models files: Add the missing gold field from a third dataset, and rename columns to match other columns
    if is_quark_model and is_the_same_model:
        # Quark models run in parallel, requires sorting. Otherwise it doesn't
        existing_preds = existing_preds.sort_values('input_jsonl_ids')

        if is_the_same_model:
            existing_preds['predicted'] = existing_preds['generated_text']
        
        # Take gold from the original jsonl
        base_model_name_or_path = model_name.replace("_quark", "").replace("_distilled", "")
        file = f"dataset/{split}_set_{base_model_name_or_path}.jsonl"
        with open(file, "r") as f:
            items = [json.loads(line) for line in f.readlines()]
        
        jsonl_df = pd.DataFrame(items)
        existing_preds['gold'] = jsonl_df['gold_summary'].tolist()

    # Add highlights (necessary for metrics)
    existing_preds = read_highlights_concatenation(existing_preds, split)

    new_preds = read_file(output_file)

    new_preds = fix_newlines(new_preds)

    # In case the predictions is still incomplete, remove values
    allow_less_values = False
    if allow_less_values:
        existing_preds = existing_preds[:len(new_preds)]
    else:
        assert len(new_preds) == existing_preds.shape[0], "File incomplete"
    existing_preds['decoding_preds'] = new_preds

    existing_preds.to_csv(f"output/{split}_combined_preds.csv", index=False)

    def calc_rouge(predictions, references):
        metric = evaluate.load('rouge', seed=42)
        rouge_scores = metric.compute(references=references, predictions=predictions)
        for rouge_key, rouge_value in rouge_scores.items():
            print(f"{rouge_key}: {'%.2f' % (rouge_value * 100)}")

    meteor_metric = evaluate.load('meteor')

    def calc_meteor(predictions, references):
        value = meteor_metric.compute(predictions=predictions, references=references)['meteor']
        print(f"meteor: {'%.2f' % (value * 100)}")

    def calc_metrics(predictions, df):
        for gold_column in ['gold', 'highlights_concatenated']:
            print(f"Comparing to column: {gold_column}")

            # Compare gold
            calc_rouge(references=df[gold_column], predictions=predictions)

            # Compare meteor
            calc_meteor(references=df[gold_column], predictions=predictions)

    # calc rouge
    print("New:")
    calc_metrics(predictions=existing_preds['decoding_preds'], df=existing_preds)

    # calc previous rouge
    if is_the_same_model:
        print("Baseline:")
        calc_metrics(predictions=existing_preds['predicted'], df=existing_preds)

    if is_the_same_model:
        compare_diffs = False
        if not compare_diffs:
            print("not comparing diffs")
        else:    
            predictions_comparison = existing_preds['predicted'].apply(lambda x: x.strip()) == existing_preds['decoding_preds'].apply(lambda x: x.strip())
            num_diffs = sum(predictions_comparison.apply(lambda x: 0 if x else 1))
            print(f"num_diffs between predictions {num_diffs} (expected 0)")
            assert num_diffs == 0, f"expected 0 diffs, found {num_diffs} diffs"
