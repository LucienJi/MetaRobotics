#!/bin/bash

task_types=("iid" "2body" "3body")
save_paths=("logs/Eval2/iid" "logs/Eval2/2body" "logs/Eval2/3body")
force_lists=("--force_list 10 15 20 25 30 35 40 45 50 55 60 65 70" "--force_list 5 10 15 20 25 30 35" "--force_list 5 10 15 20 25 30 35")

model_names=("VQ_1_head" "VQ_2_head" "VQ_4_head" "VQ_8_head" "VQ_16_head" "VQ_32_head")
n_heads=(1 2 4 8 16 32)
model_paths=("logs/VQ/Dec06_14-25-32_STG_1_head/model_10000.pt" \
"logs/VQ/Dec06_14-27-39_STG_2_head/model_10000.pt" \
"logs/VQ/Dec06_14-28-58_STG_4_head/model_10000.pt" \
"logs/VQ/Dec06_14-33-51_STG_8_head/model_10000.pt" \
"logs/VQ/Dec06_14-36-50_STG_16_head/model_10000.pt" \
"logs/VQ/Dec06_14-39-24_STG_32_head/model_10000.pt" \
)
cmd_vel_values=(0.5 1.0)

# 获取数组长度
num_task=${#task_types[@]}
num_head=${#n_heads[@]}
num_model=${#model_names[@]}

# 启动任务计数器
task_counter=0

for ((i=0; i<$num_task; i++))
do
    task_type=${task_types[$i]}
    force_list=${force_lists[$i]}
    save_path=${save_paths[$i]}

    for ((j=0; j<$num_model; j++))
    do
        model_name=${model_names[$j]}
        model_path=${model_paths[$j]}
        n_head=${n_heads[$j]}
        for cmd_vel in "${cmd_vel_values[@]}"
        do
            cmd_vel_no_dot="${cmd_vel//./}"
            new_save_path="${save_path}/v${cmd_vel_no_dot}/"
            eval_name="${task_type}_v${cmd_vel_no_dot}"

            # 计算 CUDA 设备 ID
            device_id=$((task_counter % 3 + 1))
            rl_device="cuda:${device_id}"
            sim_device="cuda:${device_id}"

            # 使用日期和时间为输出文件生成唯一的前缀
            timestamp=$(date +"%Y%m%d%H%M%S")
            output_file="output_${eval_name}_${timestamp}.out"

            nohup python eval_push_vq.py \
            --headless \
            --rl_device "cuda:${device_id}" \
            --sim_device "cuda:${device_id}" \
            --n_heads "$n_head" \
            --model_path "$model_path" \
            --eval_name "$eval_name" \
            --eval_path "$new_save_path" \
            --task_type "$task_type" \
            --model_name "$model_name" \
            --cmd_vel "$cmd_vel" \
            $force_list \
            > "$output_file" 2>&1 &

            # 更新任务计数器
            task_counter=$((task_counter + 1))

            # 可以选择在此处检查当前运行的后台任务数量
            # 并在达到一定数量时等待它们完成
            if (( task_counter % 2 == 0 )); then
                wait
            fi
        done
    done
done

# 等待所有后台任务完成
wait