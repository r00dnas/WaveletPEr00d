# for name in "bace" "bbbp" "sider" "tox21" "toxcast"
# do
# python3 aggr_results.py --dataset_name=$name --model_type="gnn_vn_gine"
# done

model=gnn_transformer_conv

#for model in "gnn_gine" "gnn_transformer_conv" "gnn_vn_gine" "gnn_vn_transformer_conv" 
for dataset_name in "bbbp" "bace" "sider" "toxcast" "tox21"
do
for model in "gnn_transformer_conv" "gnn_vn_transformer_conv"
do
python3 aggr_results.py --dataset_name=$dataset_name --model_type=$model
done
done