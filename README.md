# Standard & Adversarial Training of Wide Residual Networks

## Requirements
- PyTorch 1.4
- TorchVision 0.5

## Supported Training Methods
#### Standard Training
- Standard training with a cyclic learning rate scheduler (std_train)
- Standard training with a step learning rate scheduler (std_train_stepLR)
#### Adversarial Training
- PGD adversarial training with a cyclic learning rate scheduler (adv_train)
- Free adversarial training with a cyclic learning rate scheduler (adv_train_free)
- FGSM adversarial training with a cyclic learning rate scheduler (adv_train_fgsm)

## How to Run
#### Example:
```
python std_train_wrn.py --dataset cifar10 --model-config config/model/wrn-16-1.json --run-config config/run/std_train.json --output-path results/ --device 0" 
python adv_train_wrn.py --dataset cifar10 --model-config config/model/wrn-16-1.json --run-config config/run/adv_train.json --output-path results/ --device 0" 
```

## Training Results

#### CIFAR-10
##### Standard Training (Standard Accuracy)
| model  | std_train | std_train_stepLR |
| :---:  | :---:| :---: |
|wrn-16-1|0.9125|0.9159|
|wrn-16-2|0.9369|0.9390|
|wrn-16-4|0.9460|0.9493|
|wrn-16-8|0.9491|0.9546|
|wrn-40-1|0.9339|0.9347|
|wrn-40-2|0.9467|0.9436|
|wrn-40-4|0.9532|0.9498|
|wrn-40-8|0.9549|0.9555|

##### Adversarial Training (Adversarial Accuracy)
| model  | adv_train | adv_train_free | adv_train_fgsm |
| :---:  | :---:| :---: | :---: | 
|wrn-16-1|0.5038|0.4476|0.3190|
|wrn-16-2|0.5470|0.5002|0.3221|
|wrn-16-4|0.5677|0.5191|0.3352|
|wrn-16-8|0.5556|0.5465|0.3538|
|wrn-40-1|0.5400|0.4890|0.3500|
|wrn-40-2|0.5725|0.5218|0.4033|
|wrn-40-4|0.5701|0.5356|0.4576|
|wrn-40-8|0.5523|0.5340|0.4863|

#### CIFAR-100
##### Standard Training (Standard Accuracy)
| model  | std_train | std_train_stepLR |
| :---:  | :---:| :---: |
|wrn-16-1|0.6547|0.6734|
|wrn-16-2|0.7178|0.7157|
|wrn-16-4|0.7554|0.7632|
|wrn-16-8|0.7741|0.7845|
|wrn-40-1|0.7104|0.6927|
|wrn-40-2|0.7494|0.7388|
|wrn-40-4|0.7759|0.7744|
|wrn-40-8|0.7881|0.7873|

##### Adversarial Training (Adversarial Accuracy)
| model  | adv_train | adv_train_free | adv_train_fgsm |
| :---:  | :---:| :---: | :---: | 
|wrn-16-1|0.2457|0.2096|0.1529|
|wrn-16-2|0.2865|0.2450|0.1581|
|wrn-16-4|0.3149|0.2783|0.1398|
|wrn-16-8|0.2954|0.3107|0.1588|
|wrn-40-1|0.2850|0.2280|0.1683|
|wrn-40-2|0.3213|0.2707|0.1788|
|wrn-40-4|0.3051|0.2990|0.1908|
|wrn-40-8|0.2964|0.0038|0.2355|
