CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode texture_synthesis \
    --texture_shape -1 -1 \
    --output_dir ./results/ \
    --summary_dir ./results/log/ \
    --target_dir ./imgs/tomato.png \
    --max_iter 10000 \
    #--vgg_ckpt ./vgg19/vgg_19.ckpt