python main.py --data_root_dir ./dataset/ --pretrain_path ./dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
--gpu 0 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 256 --save kd --backbone drgat --mode teacher

python main.py --data_root_dir ./dataset/ --pretrain_path ./dataset/ogbn-arxiv-pretrain/X.all.xrt-emb.npy \
--gpu 0 --use-norm --use-labels --n-label-iters=1 --no-attn-dst --edge-drop=0.5 --input-drop=0.35 --n-layers 3 --dropout 0.8 --hid-drop 0.8 --n-hidden 256 --save kd --backbone drgat --alpha 0.95 --temp 0.7 --mode student
