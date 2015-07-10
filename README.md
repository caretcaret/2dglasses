# 2dglasses

Using modified version of CaffeNet à la http://caffe.berkeleyvision.org/gathered/examples/finetune_flickr_style.html

You will need: python, caffe (already compiled), imagemagick (convert utility)

# How to train
1. `export CAFFE=/path/to/caffe`
2. `python scripts/prepare_data.py /path/to/2d /path/to/3d` 
3. `./scripts/create_lmdb.sh`
4. `./scripts/make_mean.sh`
5. `./scripts/train_caffenet.sh`
