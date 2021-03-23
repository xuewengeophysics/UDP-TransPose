# UDP-TransPose

+ [UDP-Pose: Unbiased Data Processing for  for Human Pose Estimation](https://github.com/HuangJunJie2017/UDP-Pose)
+ [TransPose: Towards Explainable Human Pose Estimation by Transformer](https://github.com/yangsenius/TransPose)





## Getting started

### Installation

1. Install PyTorch>=1.8 and torchvision>=0.7 from the PyTorch [official website](https://pytorch.org/get-started/locally/)

   ```bash
   # for A100 GPU
   conda install pytorch torchvision cudatoolkit=11 -c pytorch-nightly
   ```

   

2. Install package dependencies. Make sure the python environment >=3.7

   ```bash
   pip install -r requirements.txt
   ```

3. Make output (training models and files) and log (tensorboard log) directories under ${POSE_ROOT} & Make libs

   ```bash
   mkdir output log
   cd ${POSE_ROOT}/lib
   make
   ```

   