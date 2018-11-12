export CUDA_VISIBLE_DEVICES=0
ulimit -n 160000
python main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/full_dataset

