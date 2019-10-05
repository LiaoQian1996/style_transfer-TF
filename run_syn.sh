CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode texture_synthesis \
    --texture_shape -1 -1 \
    --output_dir ./results/ \
    --summary_dir ./results/log/ \
    --target_dir ./imgs/tomato.png \
    --initials noise \
    --top_style_layer VGG54 \
    --max_iter 10000 \
    --save_freq 1000 \
    --summary_freq 100 \
    --decay_step 10000 \
    --learning_rate 0.1 \
    --decay_rate 0.01 \
    --W_tv 0.001
    --vgg_ckpt ./vgg19/vgg_19.ckpt