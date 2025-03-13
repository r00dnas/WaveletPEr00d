export CUDA_VISIBLE_DEVICES=0

model_type=gnn_vn
local_gnn_type=transformer_conv
scheduler=cosine_with_warmup
warmup_steps=5
num_epoch=40
dropout=0.5
freeze=1
dim=512
num_layer=4
residual=0
lr=1e-3

for i in 1 2 3 4 5
do 
python3 downstream.py --config moleculenet_tox21 --model_type=$model_type --local_gnn_type=$local_gnn_type --scheduler=$scheduler --warmup_steps=$warmup_steps --num_epoch=$num_epoch \
                      --dropout=$dropout --freeze=$freeze --atom_dim=$dim --bond_dim=$dim --num_layer=$num_layer \
                      --residual=$residual --lr=$lr
done