## Train
For the train setup, open train_setup.py and add setup key in the dictionary. All the training data will be saved into **results** folder with _setup_dict's key name
(i.e. in train_setup.py has example setup **test**. once you run training, you will see **1_32_fusion** folder in **results** which contains "checkpoints" "inference" "logs" "pretrain" "test"
              "checkpoints" : folder contains checkpoints 
              "inference"   : folder for inference. it will save full resolution depth rgbd/tofd prediction for all validation images with each validation iterations
              "logs"        : folder contains tensorboard file
              *currently "pretrain" and "test" arent be used)

### Cropping
- Cropping is for augmentation. ONLY capable if not using anything geometric (i.e. no self suprevised loss / no geometrical fusion.)
- capable with no fusion or 1/32 resolution fusion / L1 loss with or without smoothness loss ([320,320] is used)

### Fusion setup
- for none fusion (purely MD2 pipeline), set "fusion" : [False, False, False, False]
- for 1/32 resolution fusion, set "fusion" : [512, False, False, False]
- for geometrically aligned fusion, set "fusion" : [512, 128, 64, 64]

### Loss setup
- "rgb_l1" : weight for rgb_l1 loss
- "rgb_smooth" : weight for edge aware smoothness loss on rgb image
- "rgb_self : weight for self supervised photometric loss on rgb image

- "tof_l1" : weight for tof_l1 loss
- "tof_smooth" : *currently not used
- "tof_self : weight for self supervised photometric loss on rgb image

### Learning rate / Logging
- "lr" : learning rate
- "step_size" : decay every n step size
- "gamma" : decate reate with every n step size

- "checkpoint" : continue training from checkpoints. weight has to be located in _key_/checkpoints/ and only specificed by its checkpoint name
                 (i.e. chekpoints will be saved in results/_key_/checkpoints/xxxx.pth then specifing xxxx.pth is enough)
- "val_every" : validation every n step
- "save_every" : save checkpoints every n step
- "stop_at" : stop training at nth step

for training, use command
python training_loop_new.py _key_of_setup_dict_
(i.e. python training_loop_new.py full_fusion)

## Test
For the test step, open test_setup.py and add setup key in the dictionary. For testing, only images from "./testset/i-ToF" and "./testset/RGB" folder will be used and prediction will be located in both "./testset/output/rgb" and "./testset/output/tof" as 8bit image. Depth range is 0-12m. To convert into actual depth, depth has to be diveded by 255 and multiplied by 12 (if depth is read as [0,255]) or just multiplied by 12 (if depth is read as [0,1])
       
### Testsetup
- "fusion" : follows same convention as in train_setup.py
- "checkpoint" : checkpoint in testsetup needs to be filename with full path (i.e. "checkpoint":"./results/full_fusion/checkpoints/00000049_th_iter.pth")

for testing, use command
python inference.py _key_of_setup_dict_
(i.e. python inference.py full_fusion)
