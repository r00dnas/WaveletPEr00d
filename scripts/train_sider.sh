export CUDA_VISIBLE_DEVICES=0

local_gnn_type=gat
scheduler=cosine_with_warmup
warmup_steps=10
num_epoch=50

for i in 1 2 3 4 5 6 7 8 9 10
do 
python3 downstream.py --config moleculenet_sider --local_gnn_type=$local_gnn_type --scheduler=$scheduler --warmup_steps=$warmup_steps --num_epoch=$num_epoch
done