#!/bin/bash

task_types=("iid" "2body" "3body")
save_paths=("logs/Eval2/iid" "logs/Eval2/2body" "logs/Eval2/3body")
force_lists=("--force_list 10 15 20 25 30 35 40 45 50 55 60 65 70" "--force_list 5 10 15 20 25 30 35" "--force_list 5 10 15 20 25 30 35")

model_names=("VQ_4_head")
model_paths=("logs/VQ/Dec06_14-28-58_STG_4_head/model_10000.pt")
cmd_vel_values=(0.5 1.0)

# 获取数组长度
num_task=${#task_types[@]}
num_model=${#model_names[@]}

for ((i=0; i<$num_task; i++))
do
    task_type=${task_types[$i]}
    force_list=${force_lists[$i]}
    save_path=${save_paths[$i]}

    # for baseline_name in "${baseline_names[@]}"
    for ((j=0; j<$num_model; j++))
    do
        model_name=${model_names[$j]}
        model_path=${model_paths[$j]}
        for cmd_vel in "${cmd_vel_values[@]}"
        do
            cmd_vel_no_dot="${cmd_vel//./}"
            new_save_path="${save_path}v${cmd_vel_no_dot}/"
            eval_name="${task_type}v${cmd_vel_no_dot}"
            nohup python eval_push_vq.py \
            --headless \
            --rl_device cuda:2 \
            --sim_device cuda:2 \
            --model_path "$model_path" \
            --eval_name "$eval_name" \
            --eval_path "$new_save_path" \
            --task_type "$task_type" \
            --model_name "$model_name" \
            --cmd_vel "$cmd_vel" \
            $force_list \
            > output_vq.out
        done
    done
done