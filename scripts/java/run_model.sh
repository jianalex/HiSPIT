#!/bin/bash

# 定义模型名称数组
models=("0511_lr_decay_08" "0512_lr_decay_085" "0515_lr_decay_09" "0516_lr_decay_095" )

# 遍历每个模型名称
for model_name in "${models[@]}"
do
    # 使用 transformer.sh 来训练模型
    #echo "Training $model_name..."
    bash transformer.sh 0 "$model_name"
    #echo "$model_name training completed."
    #if [ -f ~/alexliu/my/tmp/"$model_name".txt ]; then
    #    echo "File exists."
    #else
    #    echo "File does not exist."
    #fi
done

echo "All models have been tested."

python /home/selab/alexliu/coco-caption/myEval.py