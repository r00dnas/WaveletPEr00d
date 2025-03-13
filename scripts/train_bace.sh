export CUDA_VISIBLE_DEVICES=0

model_type=GT
local_gnn_type=gine
scheduler=cosine_with_warmup
warmup_steps=10
num_epoch=50

for i in 1 2 3 4 5
do 
python3 downstream.py --config moleculenet_bace --model_type=$model_type --local_gnn_type=$local_gnn_type --scheduler=$scheduler --warmup_steps=$warmup_steps --num_epoch=$num_epoch
done