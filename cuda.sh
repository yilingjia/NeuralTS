python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv diag --dataset shuttle
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mushroom
python3 train.py --encoding multi --learner neural --style ucb --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner neural --style ts --nu 0.00001 --lamdba 0.00001 --inv diag --dataset shuttle
python3 train.py --encoding multi --learner neural --style ts --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mushroom
python3 train.py --encoding multi --learner neural --style ts --nu 0.00001 --lamdba 0.00001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner linear --style ucb --nu 0.001 --lamdba 0.001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner linear --style ts --nu 0.001 --lamdba 0.001 --inv diag --dataset mnist
python3 train.py --encoding multi --learner linear --style ucb --nu 0.001 --lamdba 0.001 --inv diag --dataset isolet
python3 train.py --encoding multi --learner linear --style ts --nu 0.001 --lamdba 0.001 --inv diag --dataset isolet