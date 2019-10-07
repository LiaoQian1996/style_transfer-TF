CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode style_transfer \
    --output_dir ./results/ \
    --target_dir ./imgs/starry-night.png \
    --content_dir ./imgs/tubingen.png \
    --top_style_layer VGG54 \
    --max_iter 50 \
    --W_tv 0.001 \
    --W_content 1e-6 \
    --vgg_ckpt ./vgg19/vgg_19.ckpt