# An implementation of RECNN "On Training Deep 3D CNN Models with Dependent Samples in Neuroimaging"
This is an implementation of RECNN as decribed in the paper [On Training Deep 3D CNN Models with Dependent Samples in Neuroimaging](https://link.springer.com/chapter/10.1007/978-3-030-20351-1_8)

# Training Recipe
1. number of epochs:50
2. learning rate schedule: step decay, initial lr = 0.0001
3. weight decay: 4e-5

# Usage 
python recnn.py --gpus=0,1 --workers=16 --classes=2 --batch-size=8 --model=recnn --save=RECNN --learning_rate=0.0001 --h5dir=../datasets --seed=97 --log-interval=5

# Reference
```
@inproceedings{xiong2019training,
  title={On Training Deep 3D CNN Models with Dependent Samples in Neuroimaging},
  author={Xiong, Yunyang and Kim, Hyunwoo J and Tangirala, Bhargav and Mehta, Ronak and Johnson, Sterling C and Singh, Vikas},
  booktitle={International Conference on Information Processing in Medical Imaging},
  pages={99--111},
  year={2019},
  organization={Springer}
}
```
