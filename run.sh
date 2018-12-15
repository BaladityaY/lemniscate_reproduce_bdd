export CUDA_VISIBLE_DEVICES=0 
ulimit -n 160000
python main.py -j 0 --gpu 0 --batch-size 32 /home/bala/for_bdd_training/smaller_dataset --resume /home/bala/lemniscate_reproduce_exps/lemniscate_reproduce_bdd_fr4/checkpoint_epoch03.pth.tar
