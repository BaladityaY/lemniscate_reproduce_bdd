export CUDA_VISIBLE_DEVICES=1
ulimit -n 160000
#python main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/smaller_dataset
#python main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/full_dataset
#python main.py -j 0 --gpu 0 --batch-size 32 /home/sascha/for_bdd_training/tiny_test_set
python main.py -j 0 --gpu 0 --batch-size 32 --train-data /data/dataset/full_train_set.hdf5 --val-data /data/dataset/full_val_set.hdf5
#python main.py -j 0 --gpu 0 --batch-size 32 --train-data /data/dataset/tiny_train_set.hdf5 --val-data /data/dataset/tiny_val_set.hdf5
