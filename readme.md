# style_transfer-TF

This is a Tensorflow implementation of papers [Texture Synthesis Using Convolutional Neural](http://papers.nips.cc/paper/5633-texture-synthesis-using-convolutional-neural-networks.pdf) and [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576) by Leon A. Gatys, Alexander S.Ecker, and Matthias Bethge.

The former presents an texture synthesis algorithm by modeling the statistics feature extracted from Convolutional Neural Networks. The latter algorithm combines the texture feature, that is , the style, of one so called style image with the content of another content image, and generated a style transfer image. 

### Examples of synthesis texture

<table>
	<tr>
		<td><center> tomato2pepper </center></td>
		<td><center> pepper2tomato </center></td>
	</tr>
	<tr>
		<td>
			<center><img src = "./results/tomato2pepper.gif"></center>
		</td>
		<td>
			<center><img src = "./results/pepper2tomato.gif"></center>
		</td>
	</tr>
</table>

### Example of Style Transfer

<table>
	<tr>
        <center> Content Image Tubingen </center>
        <center><img src="./imgs/tubingen.png" width="600px"></center>
    </tr>
	</br>
    <tr>
        <center> Style Image Starry Night </center>
    	<center><img src="./imgs/starry-night.png" width="600px"><center>
    </tr>
    </br>
    <tr>
        <center> Stylized Image </center>
    	<center><img src="./results/VGG54_1000_1.1148e-05_5.5795e+01_starry_night_tubingen.png" width="600px"><center>
    </tr>
</table>


### Dependency
* python3.5
* tensoflow (tested on r1.4)
* VGG19 model weights download from the [TF-slim models](http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz) 
* The code is tested on:
	* Ubuntu 16.04 LTS with CPU architecture x86_64 + Nvidia GeForce GTX 1080

### Recommended
* Ubuntu 16.04 with TensorFlow GPU edition

### Getting started 
Denote the directory of this repository as ```./style_transfer-TF/``` 

* #### Run texture synthesis demo  to check configuration

```bash
# clone the repository from github
git clone https://github.com/LiaoQian1996/style_transfer-TF.git
cd $style_transfer-TF/

# download the vgg19 model weights from 
# http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz
# to style_transfer-TF/vgg19/

# run the texture synthesis demo
sh demo.sh
# the result can be viewed at $./results/
```
* #### Synthesis your texture image
```bash
cd $style_transfer-TF/

# put your own png images in $style_transfer-TF/imgs/
```
modify the parameters in ``` run_syn.sh``` 

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --task_mode texture_synthesis \
    --texture_shape 512 512 \ # set the synthesized texture shape, which will be same as sample's if set as [-1, -1]  
    --output_dir ./results/ \
    --summary_dir ./results/log/ \
    --target_dir ./imgs/your_own_image_name.png \  # your own image, should be .png format and RGB mode 
    --content_dir ./imgs/tomato.png \ # use this image to initialize the systhesized image, --initials should be set as content \
    --initials content \ # noise (option)
    --top_style_layer VGG54 \ # VGG 11 21 31 41 51 54
    --max_iter 10000 \
    --save_freq 1000 \ 
    --summary_freq 100 \
    --decay_step 10000 \
    --learning_rate 0.1 \ 
    --decay_rate 0.01 \ # learning_rate will exponential decay 
    --W_tv 0.001 \ # weight of total variation loss
    --vgg_ckpt ./vgg19/vgg_19.ckpt
```
then run the shell script
```
sh run_syn.sh
```
* #### Style Transfer
```bash
cd $style_transfer-TF/

# put your own content and style images (.png) in $style_transfer-TF/imgs/
```
modify the parameters in ``` run_transfer.sh``` 

```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --task_mode style_transfer \
    --output_dir ./results/ \
    --summary_dir ./results/log/ \
    --target_dir ./imgs/starry-night.png \         # path of style image
    --content_dir ./imgs/tubingen.png \           # path of content image
    --initials content \                                       # initial synthesized image, noise or content 
    --top_style_layer VGG54 \       
    --max_iter 1000 \
    --save_freq 200 \
    --save_step 10 20 50 \
    --display_freq 10 \
    --summary_freq 100 \
    --learning_rate 0.1 \
    --beta1 0.5 \
    --decay_rate 1.0 \
    --W_tv 0.001 \
    --W_content 1e-6 \
    #--vgg_ckpt ./vgg19/vgg_19.ckpt
```
then run the shell script
```
sh run_transfer.sh
```

The result can be viewed at $style_transfer-TF/results/ and the output images will be named as  "top-style-layer_iteration-step_style-loss_tv-loss_image-name.png"

Launch tensorboard to monitor the optimization process
```bash
# Launch the tensorboard
cd ./style_transfer-TF/results/
tensorboard --logdir=log
# Then load tensorboard in your browser
```
