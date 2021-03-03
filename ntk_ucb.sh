python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv full --dataset adult
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv full --dataset covertype
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv full --dataset MagicTelescope
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv full --dataset mnist
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv full --dataset mushroom
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv full --dataset shuttle

time CUDA_VISIBLE_DEVICE=1 python train.py --lamdba 0.0001 --nu 0.0001 --dataset mushroom