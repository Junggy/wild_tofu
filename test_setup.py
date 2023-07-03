_setup_dict= {
    "full_fusion":{
        "base": "./testset",
        "val_batch": 1,
        "dmin": 0.3,
        "dmax": 12,
        "filtered": True,

        "fusion": [512, 128, 64, 64],

        "checkpoint":"./results/full_fusion/checkpoints/00000049_th_iter.pth",
    }
}