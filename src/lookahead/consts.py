model_name_to_path = {
    "flan": "models/flan-t5-large/finetune_flan_t5_large_on_pretrain_cnndm_duc/checkpoint-1600",
    "flan_distilled": "models/flan-t5-large/finetune_on_pretrain_cnndm_duc_on_distilled_GPT4/checkpoint-2000",
    "flan_quark": "models/flan-t5-large/best_Flan_t5_large_quark_reward_type_concats_P_reward_type_highlights_precision_rougeL_R_reward_type_highlights_recall_rougeL/ckp_14000",
    "flan_distilled_quark": "models/flan-t5-large/Flan_t5_large_distilled/ckp_5500",
    "flan_quark_init": "models/flan-t5-large/init_models/checkpoint-1600",
    "flan_distilled_quark_init": "models/flan-t5-large/init_models/finetune_on_pretrain_cnndm_duc_on_distilled_GPT4/checkpoint-2000",
    "led": "models/led/finetune_led_large_on_pretrain_cnndm_duc/checkpoint-300",
    "led_distilled": "models/led/finetune_led_large_on_pretrain_cnndm_duc_distilled/checkpoint-1600",
    "led_quark": "models/led/LED_large_batch2_alternate_reward_type_concats_max_src_len_1400_P_reward_type_highlights_precision_rougeL_R_reward_type_highlights_recall_rougeL_no_rpt_ngrams_3_beams_2_len_penalty_2_0_sample_top_p_0.9_sampleEval/ckp_500",
    "led_quark_init": "models/led/init_models/finetune_led_large_on_pretrain_cnndm_duc/checkpoint-300",
    "led_distilled_quark": "models/led/LED_large_distilled_batch2_max_src_len_1400_P_reward_type_highlights_precision_rougeL_R_reward_type_highlights_recall_rougeL/ckp_15000",
    "led_distilled_quark_init": "models/led/init_models/finetune_led_large_on_pretrain_cnndm_duc_distilled/checkpoint-1600",
    "distilled": "models/flan-t5-large/flan_t5_large_on_distilled_GPT4_max_target_length_512/checkpoint-1400", # TODO: Check which one is this if we want to use these results
}