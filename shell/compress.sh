compression_path="algorithms/compression"
nets_path="${compression_path}/nets"
model_path="${nets_path}/yolov5-7.0"


dataset=$1
model=$2
prune_method=$3
quan_method=$4
ft_lr=$5
ft_bs=$6
ft_epochs=$7
prune_sparisity=$8
gpus=$9
input_path=${10}
output_path=${11}
dataset_path=${12}


source ~/conda3/etc/profile.d/conda.sh
# conda activate yolo

if [ $prune_method != 'null' ] && [ $quan_method == 'null' ] # prune
then
    conda activate yolo
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/prune_yolov5.py \
            --device=${gpus} \
            --pre_weights=${input_path}/best.pt \
            --pruner=${prune_method} \
            --output_dir=${output_path} \
            --calc_initial_yaml \
            --calc_final_yaml \

    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/prune_train.py \
            --device=${gpus} \
            --weights=${output_path}/pruned_yolov5s_voc_fpgms_0.15.pt \
            --batch-size=${ft_bs} \
            --log_dir=${output_path} \
            --epochs=${ft_epochs} \
            --calc_final_yaml \


    conda deactivate

elif [ $prune_method == 'null' ] && [ $quan_method != 'null' ] # quan
then
    conda activate yolo-quant

    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/export.py \
            --weights=${input_path}/best.pt \
            --output_dir=${output_path} \
            --calc_initial_yaml \
            --calc_final_yaml \
            --half


    conda deactivate

elif [ $prune_method != 'null' ] && [ $quan_method != 'null' ] # prune and quan
then
    conda activate yolo
    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/prune_yolov5.py \
            --device=${gpus} \
            --pre_weights=${input_path}/best.pt \
            --pruner=${prune_method} \
            --output_dir=${output_path} \
            --calc_initial_yaml \
            --calc_final_yaml \
            --val_batch_size 1

    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/prune_train.py \
            --device=${gpus} \
            --weights=${output_path}/pruned_yolov5s_voc_fpgms_0.15.pt \
            --batch-size=${ft_bs} \
            --log_dir=${output_path} \
            --epochs=${ft_epochs} \
            # --calc_final_yaml \


    conda deactivate

    conda activate yolo-quant

    mkdir -p ${output_path}
    CUDA_VISIBLE_DEVICES=${gpus} \
        python ${model_path}/export.py \
            --weights=${output_path}/prune_trained_pruned_yolov5s_voc_fpgms_0.15.pt \
            --output_dir=${output_path} \
            --calc_final_yaml \
            --half
            # --calc_initial_yaml \


    conda deactivate

fi

conda deactivate