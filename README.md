# frozen-graph-training
This code offers the possibility of training deep learning image classiers in TensorFlow 2 using VGG-16 trained on ImageNet as a base network. Data augmentation and resampling are used by default. The final model is finally stored as frozen graph and can be used for inferencing in TensorFlow 1. The example images in the image folders are taken from [1].

## Setup
Clone the repository:
```sh
git clone https://github.com/lkreiskoether/frozen-graph-training.git
```

## General usage
```sh
python grad-cam-analysis.py <path to folder containing image folders> <path folder for model storing> '[<x-dim images>,<y-dim images>]' <number of epochs> <learning rate> <batch size> <data split for validation & testing> <data split for testing based on data split for validation & testing> 
```

## Example usage
```sh
python fg_training.py mvtec mvtec '[100,100]' 3 0.0001 4 0.5 0.5
```

## Acknowledgements
Thanks to https://leimao.github.io/blog/Save-Load-Inference-From-TF2-Frozen-Graph/ for providing the code for storing frozen graph models in TensorFlow 2.

## References
[1] P. Bergmann, M. Fauser, D. Sattlegger, and C. Steger, “MVTEC ad-A comprehensive real-world dataset for unsupervised anomaly detection,” in Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition, 2019, vol. 2019-June, pp. 9584–9592, doi: 10.1109/CVPR.2019.00982.
