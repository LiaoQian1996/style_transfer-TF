CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode style_transfer \
    --output_dir ./results/ \
    --summary_dir ./results/log/ \
    --target_dir ./imgs/starry-night.png \
    --content_dir ./imgs/zoufangyue.png \
    --initials content \
    --top_style_layer VGG54 \
    --max_iter 1000 \
    --save_freq 100 \
    --save_step 10 20 50 \
    --display_freq 10 \
    --summary_freq 100 \
    --learning_rate 0.1 \
    --beta1 0.5 \
    --decay_rate 1.0 \
    --W_tv 0.001 \
    --W_content 1e-6 \
    #--vgg_ckpt ./vgg19/vgg_19.ckpt