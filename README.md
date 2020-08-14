# kaggle-wheat-arutema47
grave for my wheat challenge

# EfficientDet train script

* training-efficientdet-v2.ipynb

I used this script to train my final model.

With psuedo labeling, this model can achieve LB/PB 0.761/0.707.
Ensemble further boosts to LB 0.767.

* eval-efficientdet-sources.ipynb

  I used this script to evaluate mAP by data sources.

  Since it is hard to tell model performance with concatenated mAP, data source split tells more about it.

# Centernet

I mainly worked on Centernet training for this challenge.

The implementation is based on [camaro's repo](https://github.com/bamps53/kaggle-autonomous-driving2019).

Although worse than Effdet for wheat, Centernet is much more easier to customize.

* config/3x3_traincrop_mixup.yaml

should be the best model setup which uses rx101 for backbone and fpn.

* centernet_train.ipynb

  Is the main training script.
  
