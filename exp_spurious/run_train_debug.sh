# Train spuriously correlated models
python3 train.py --dataset="bear-bird-cat-dog-elephant:dog(chair)" --device 0
python3 train.py --dataset="bear-bird-cat-dog-elephant:dog(water)" --device 0
python3 train.py --dataset="bear-bird-cat-dog-elephant:cat(book)" --device 0
python3 train.py --dataset="bear-bird-cat-dog-elephant:bird(sand)" --device 0
python3 train.py --dataset="bear-bird-cat-dog-elephant:cat(keyboard)" --device 0

python evaluate_cce.py --device 0 --concept-bank="ckpts/resnet18_bank.pkl" --dataset "bear-bird-cat-dog-elephant:dog(chair)"
python evaluate_cce.py --device 0 --concept-bank="ckpts/resnet18_bank.pkl" --dataset "bear-bird-cat-dog-elephant:dog(water)"
python evaluate_cce.py --device 0 --concept-bank="ckpts/resnet18_bank.pkl" --dataset "bear-bird-cat-dog-elephant:cat(book)"
python evaluate_cce.py --device 0 --concept-bank="ckpts/resnet18_bank.pkl" --dataset "bear-bird-cat-dog-elephant:bird(sand)"
python evaluate_cce.py --device 0 --concept-bank="ckpts/resnet18_bank.pkl" --dataset "bear-bird-cat-dog-elephant:cat(keyboard)"

cd ..

python -m scripts.find_spurious_attributions --device 0 --dataset "bear-bird-cat-dog-elephant:dog(chair)"
python -m scripts.find_spurious_attributions --device 0 --dataset "bear-bird-cat-dog-elephant:dog(water)"
python -m scripts.find_spurious_attributions --device 0 --dataset "bear-bird-cat-dog-elephant:cat(book)"
python -m scripts.find_spurious_attributions --device 0 --dataset "bear-bird-cat-dog-elephant:bird(sand)"
python -m scripts.find_spurious_attributions --device 0 --dataset "bear-bird-cat-dog-elephant:cat(keyboard)"