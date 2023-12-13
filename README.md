# MoMo Strategy: Learn More from More Mistakes
This repository contains the implementation codes to our MoMo experiments. 

This research investigates the selection and utilization of misclassified training samples to enhance the accuracy of CNNs where the dataset is long-tail distributed.

Our experimental results on a subset of the current largest plant dataset, PlantCLEF 2023, demonstrate an increase of 1%-2% in the overall validation accuracy and a 2%-5% increase in the tail class identification.

These findings emphasize the significance of adding more misclassified samples into training, encouraging researchers to rethink the sampling strategies before implementing more complex and robust network architectures and modules.

![MoMo Strategy](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/static/overview.png)

## Research article
MoMo Strategy: Learn More from More Mistakes <br>
[https://doi.org/10.1109/APSIPAASC58517.2023.10317346](https://doi.org/10.1109/APSIPAASC58517.2023.10317346)

## Requirements
- TensorFlow 1.12
- [TensorFlow-Slim library](https://github.com/tensorflow/models/tree/r1.12.0/research/slim)
- [Pre-trained models (Inception-ResNet-v2)](https://github.com/tensorflow/models/tree/r1.12.0/research/slim#pre-trained-models)

## Dataset
[PlantCLEF2023](https://www.aicrowd.com/challenges/lifeclef-2022-23-plant)

## Scripts
**Training lists**
- Model 1
  - [train_model.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/train_model.py)
  - [network_module_baseline.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/network_module_baseline.py)
  - [database_module.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/database_module.py)

- Model 2
  - [train_model2_balanced_50.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/train_model2_balanced_50.py)
  - [network_module.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/network_module.py)
  - [database_module_balanced_50.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/database_module_balanced_50.py)

 - Model 3
   - [train_model3_balanced_70.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/train_model3_balanced_70.py)
   - [network_module.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/network_module.py)
   - [database_module_balanced_70.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/database_module_balanced_70.py)

 - Model 4
   - [train_model4_unbalanced_50.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/train_model4_unbalanced_50.py)
   - [network_module.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/network_module.py)
   - [database_module_unbalanced_50.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/database_module_unbalanced_50.py)

 - Model 5
   - [train_model5_unbalanced_70.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/train_model5_unbalanced_70.py)
   - [network_module.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/network_module.py)
   - [database_module_unbalanced_70.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/database_module_unbalanced_70.py)

**Testing lists**
- Get overall results
  - [validate_model.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/validate_model.py)
- Get head, middle, and tail classes results
  - [validate_model_head_middle_tail.py](https://github.com/NeuonAI/momo_strategy/blob/4d9e9bef71e296a370a4043a5c6ce8c85bddf9cd/scripts/validate_model_head_middle_tail.py)








