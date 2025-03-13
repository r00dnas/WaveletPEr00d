export CUDA_VISIBLE_DEVICES=2

dataset=MNIST
num_layer=3
model_type=GT
local_gnn_type=gated_gcn
atom_dim=64
bond_dim=64
scheduler=cosine_with_warmup
warmup_steps=5 #only used when scheduler = cosine_with_warmup
num_epoch=100
lr=1e-3
batch_size=512
ckpt_pos_encoder_path=/cm/shared/khangnn4/WavePE/ckpts/PCBA_debug_1/PCBA_epoch=99_train_loss=0.010_val_loss=0.011_val_best_loss=0.011.ckpt
dropout=0.0
attn_dropout=0.0

for seed in 1 
do 
python3 train_image.py --dataset=$dataset --model_type=$model_type --num_layer=$num_layer \
                      --local_gnn_type=$local_gnn_type --atom_dim=$atom_dim --bond_dim=$bond_dim \
                      --scheduler=$scheduler --warmup_steps=$warmup_steps --num_epoch=$num_epoch \
                      --lr=$lr  --learnable --batch_size=$batch_size \
                      --ckpt_pos_encoder_path=$ckpt_pos_encoder_path --dropout=$dropout --attn_dropout=$attn_dropout \
                      --not_use_full_graph --no_residual 
done