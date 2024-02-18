# <h2 align="center"> Don’t Add, don’t Miss: Effective Content Preserving Generation from Pre-Selected Text Spans </h2>

Repository for our EMNLP 2023 findings paper "[Don’t Add, don’t Miss: Effective Content Preserving Generation from Pre-Selected Text Spans](https://aclanthology.org/2023.findings-emnlp.852/)"

In this repository, we include our 3 techniques to improve the Controlled Text Reduction (CTR) task: {C}ontrolled decoding, {D}istillation from GPT-4 and {R}einforcement Learning (CDR).

## Download Dataset
To download the original Controlled Text Reduction dataset, follow the instructions in [link](https://github.com/lovodkin93/Controlled_Text_Reduction), and save it under the `data`. \
For the GPT-4 distilled training data, download it from [GPT4-distilled data](https://drive.google.com/file/d/19j7w0A3XgBBvTsy8RNOevnpAD93yvPK3/view?usp=sharing), unzip it and save it under `data`.

## Supervised Training Experiments
To train the Flan-T5 model on the original CTR dataset, run:
```
python -m src.run_experiments configs/train/flan_t5_large/pretrain_cnndm_duc_flan_t5_large.json
```
To first finetune the model on the combination of DUC- and CNNDM-derived dataset.\
Then, update the `model_name_or_path` parameter under `configs/train/flan_t5_large/finetune_flan_t5_large_on_pretrained_CNNDM_duc_full.json` to point to the best checkpoint from the previous run, and then run:
```
python -m src.run_experiments configs/train/flan_t5_large/finetune_flan_t5_large_on_pretrained_CNNDM_duc_full.json
```

* to do the same experiments with LED, simply replace the config files paths' `flan_t5_large` subdir with `LED_large`.

## Supervised Training Experiments - GPT4-distilled Dataset
To perform the supervised training experiments with the GPT4-distilled dataset, replace in the previous config paths `flan_t5_large` with `distilled_flan_t5_large` and `LED_large` with `distilled_LED_large`.



## Best Model Weights
You can download the weights of the best variant in:
[best model weights](https://drive.google.com/drive/folders/11k_BTiXD6ItjEhN4wp267HRjg1euttL7?usp=sharing)
