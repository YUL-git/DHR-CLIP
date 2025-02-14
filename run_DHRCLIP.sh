#------------------------------------Proposed method------------------------------------#
device=0

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
dataset=('visa')
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do

        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_DHRCLIP_mvtec
        save_dir=./checkpoints/${base_dir}/

        CUDA_VISIBLE_DEVICES=${device} python train_DHRCLIP.py --dataset mvtec --train_data_path /home/jiyul/SPS_JY/12.1/GlocalCLIP/data/mvtec \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 336  --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --dpam 20
    wait

    for z in "${!dataset[@]}"; do

        result_dir=./results/${base_dir}/zero_shot_${dataset[z]}/
        mkdir -p ${result_dir}

        CUDA_VISIBLE_DEVICES=${device} python test_DHRCLIP.py --dataset ${dataset[z]} \
        --data_path /home/jiyul/SPS_JY/12.1/GlocalCLIP/data/${dataset[z]} --save_path ${result_dir} --checkpoint_path ${save_dir}epoch_15.pth \
        --features_list 6 12 18 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --sigma 8 --dpam 20 --metrics pixel-level
        wait
        done
    done
done

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## train on the VisA dataset
        base_dir=${depth[i]}_${n_ctx[j]}_${t_n_ctx[0]}_DHRCLIP_visa
        save_dir=./checkpoints/${base_dir}/
        
        CUDA_VISIBLE_DEVICES=${device} python train_DHRCLIP.py --dataset visa --train_data_path /home/jiyul/SPS_JY/12.1/GlocalCLIP/data/visa \
        --save_path ${save_dir} \
        --features_list 6 12 18 24 --image_size 336  --batch_size 8 --print_freq 1 \
        --epoch 15 --save_freq 1 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --dpam 20
        wait
        
        CUDA_VISIBLE_DEVICES=${device} python test_DHRCLIP.py --dataset mvtec \
        --data_path /home/jiyul/SPS_JY/12.1/GlocalCLIP/data/mvtec --save_path ./results/${base_dir}/zero_shot --checkpoint_path ${save_dir}epoch_15.pth \
        --dpam 20 --features_list 6 12 18 24 --image_size 336 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]} --metrics pixel-level
        wait
    done
done