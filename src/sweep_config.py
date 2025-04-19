## Contains configuration for sweep using wandb


# Filter and kernel size descriptions for reference
filters_des = {
    "same_8": [8, 8, 8, 8, 8],
    "same_16": [16, 16, 16, 16, 16],
    "same_32": [32, 32, 32, 32, 32],
    "same_64": [64, 64, 64, 64, 64],
    "increase_16_128": [16, 32, 64, 128, 128],
    "decrease_128_16": [128, 128, 64, 32, 16],
    "mixed": [16, 32, 64, 32, 16],
}
kernels_des = {
    "same_3": [3, 3, 3, 3, 3],
    "same_5": [5, 5, 5, 5, 5],
    "mix_3_5": [3, 3, 5, 5, 5],
}

sweep_config = {
    "method": "bayes",
    "metric": {"name": "val_acc", "goal": "maximize"},
    "early_terminate": {"type": "hyperband", "min_iter": 3},
    "parameters": {
        "lr": {"min": 1e-5, "max": 1e-4},
        "batch_size": {"values": [16, 32]},
        "filters": {
            "values": [
                filters_des["same_8"],
                filters_des["same_16"],
                filters_des["same_32"],
                filters_des["same_64"],
                filters_des["increase_16_128"],
                filters_des["decrease_128_16"],
                filters_des["mixed"],
            ]
        },
        "kernel": {
            "values": [
                kernels_des["same_3"],
                kernels_des["same_5"],
                kernels_des["same_7"],
                kernels_des["mix_3_5"],
            ]
        },
        "pool_kernel": {"values": [[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [2, 2, 2, 3, 3]]},
        "pool_stride": {"values": [[1, 1, 1, 1, 1], [1, 1, 1, 2, 2], [1, 1, 1, 1, 2]]},
        "batchnorm": {"values": [True, False]},
        "activation": {"values": ["relu", "gelu", "mish", "swish", "selu"]},
        "augmentation": {"values": [True, False]},
        "dropout": {"min": 0.3, "max": 0.4},
        "ffn_size": {"values": [128, 256]},
        "epochs": {"values": [5]},
        "optim": {
            "values": [
                "adam",
                "sgd",
            ]
        },
    },
}
