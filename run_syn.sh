CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode texture_synthesis \
    --texture_shape -1 -1 \
    --output_dir ./results/ \
    --target_dir ./imgs/tomato.png \
    --initials noise \
    --top_style_layer VGG54 \
    --max_iter 500 \
    --W_tv 0.001 \
    --vgg_ckpt ./vgg19/vgg_19.ckpt