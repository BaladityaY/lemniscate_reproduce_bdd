export CUDA_VISIBLE_DEVICES=0
ulimit -n 160000
#python main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/smaller_dataset
#python main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/full_dataset
python3 main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/tiny_test_set

