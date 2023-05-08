config_file=$1
session=$2
cuda_visible_device=$3

WANDB_PROJECT="instructions_finetuning_CTR"
cuda_deviced_separated=$(echo "$cuda_visible_device" | tr ',' ' ')
nproc_per_node=$(echo "$cuda_deviced_separated" | wc -w)


command="export WANDB_PROJECT=${WANDB_PROJECT} && conda activate instruction_tuning_37 && cd ~/controlled_reduction/instruction_finetuning && PYTHONPATH=/home/nlp/sloboda1/controlled_reduction/instruction_finetuning CUDA_VISIBLE_DEVICES=${cuda_visible_device} python -m torch.distributed.launch  ${accelerator_config_file} --nproc_per_node=${nproc_per_node} src/run_experiments.py ${config_file}" 
echo $command

tmux new-session -d -s $session
tmux send-keys -t $session "$command" C-m
tmux attach-session -t $session

