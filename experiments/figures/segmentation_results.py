# (Frozen, Pretrained)

RESULTS = {
    "F-Actin": {
        "ResNet18_from-scratch": {
            "IoU_Rings": 0.1756,
            "IoU_Fibers": 0.2667,
            "AUPR_Rings": 0.4598,
            "AUPR_Fibers": 0.2667},
        "MAE-small_from-scratch": {
            "IoU_Rings": 0.3551,
            "IoU_Fibers": 0.5191,
            "AUPR_Rings": 0.5454,
            "AUPR_Fibers": 0.7316},
        "ResNet18_HPA": {
            "IoU_Rings": (0.1991, 0.2691), 
            "IoU_Fibers": (0.3765, 0.3392), 
            "AUPR_Rings": (0.4724, 0.4914), 
            "AUPR_Fibers": (0.6088, 0.6251)},
        "MAE-small_HPA": {
            "IoU_Rings": (0.4355, 0.3544), 
            "IoU_Fibers": (0.5465, 0.5048), 
            "AUPR_Rings": (0.5851, 0.5177), 
            "AUPR_Fibers": (0.7449, 0.7222)},
        "ResNet18_ImageNet": {
            "IoU_Rings": (0.1969, 0.1629), 
            "IoU_Fibers": (0.3552, 0.2969), 
            "AUPR_Rings": (0.4714, 0.4349), 
            "AUPR_Fibers": (0.5986, 0.6018)},
        "MAE-small_ImageNet": {
            "IoU_Rings": (0.3719, 0.2593), 
            "IoU_Fibers": (0.5205, 0.4474), 
            "AUPR_Rings": (0.5258, 0.5197), 
            "AUPR_Fibers": (0.7354, 0.7134)},
        "ResNet18_STED": {
            "IoU_Rings": (0.2511, 0.2078), 
            "IoU_Fibers": (0.4119, 0.4843), 
            "AUPR_Rings": (0.5804, 0.5887),
             "AUPR_Fibers": (0.7485, 0.7672)},
        "MAE-small_STED": {
            "IoU_Rings": (0.4166, 0.4031), 
            "IoU_Fibers": (0.6085, 0.4505), 
            "AUPR_Rings": (0.6078, 0.5435),
            "AUPR_Fibers": (0.7666, 0.7255)},
        "MAE-small-5%_from-scratch": {
            "IoU_Rings": 0.1745,
            "IoU_Fibers": 0.4288,
            "AUPR_Rings": 0.4780,
            "AUPR_Fibers": 0.6712},
        "MAE-small-5%_HPA": {
            "IoU_Rings": (0.3561, 0.1601), 
            "IoU_Fibers": (0.5446, 0.3876), 
            "AUPR_Rings": (0.5600, 0.4489), 
            "AUPR_Fibers": (0.7293, 0.6665)},
        "MAE-small-5%_ImageNet": {
            "IoU_Rings": (0.0225, 0.0414), 
            "IoU_Fibers": (0.4256, 0.4849), 
            "AUPR_Rings": (0.4491, 0.4534), 
            "AUPR_Fibers": (0.6706, 0.6716)},
        "MAE-small-5%_STED": {
            "IoU_Rings": (0.3324, 0.0431), 
            "IoU_Fibers": (0.5784, 0.5764), 
            "AUPR_Rings": (0.5513, 0.4596),
            "AUPR_Fibers": (0.7545, 0.6735)},
        "MAE-tiny_from-scratch": {
            "IoU_Rings": 0.3771,
            "IoU_Fibers": 0.4785,
            "AUPR_Rings": 0.5397,
            "AUPR_Fibers": 0.7268},
        "MAE-tiny_HPA": {
            "IoU_Rings": (0.3974, 0.3349), 
            "IoU_Fibers": (0.5724, 0.4543), 
            "AUPR_Rings": (0.5748, 0.5186), 
            "AUPR_Fibers": (0.7448, 0.7326)},
        "MAE-tiny_ImageNet": {
            "IoU_Rings": (0.3547, 0.3430), 
            "IoU_Fibers": (0.5149, 0.5185), 
            "AUPR_Rings": (0.5327, 0.5117), 
            "AUPR_Fibers": (0.7359, 0.7227)},
        "MAE-tiny_STED": {
            "IoU_Rings": (0.4163, 0.3855), 
            "IoU_Fibers": (0.6051, 0.5419), 
            "AUPR_Rings": (0.6133, 0.5532),
            "AUPR_Fibers": (0.7613, 0.7412)},
    },
    "Footprocess": {
            "ResNet18_from-scratch": {
            "IoU_FP": 0.4771,
            "IoU_SD": 0.5196,
            "AUPR_FP": 0.6896,
            "AUPR_SD": 0.7240},
        "MAE-small_from-scratch": {
            "IoU_FP": 0.3061,
            "IoU_SD": 0.4663,
            "AUPR_FP": 0.4605,
            "AUPR_SD": 0.6770},
        "ResNet18_HPA": {
            "IoU_FP": (0.4389, 0.4828), 
            "IoU_SD": (0.5011, 0.5270), 
            "AUPR_FP": (0.6473, 0.6954), 
            "AUPR_SD": (0.7201, 0.7326)},
        "MAE-small_HPA": {
            "IoU_FP": (0.4560, 0.3213), 
            "IoU_SD": (0.5212, 0.4784), 
            "AUPR_FP": (0.6756, 0.5083), 
            "AUPR_SD": (0.7266, 0.6838)},
        "ResNet18_ImageNet": {
            "IoU_FP": (0.3785, 0.4746), 
            "IoU_SD": (0.4799, 0.5183), 
            "AUPR_FP": (0.5778, 0.6902), 
            "AUPR_SD": (0.6897, 0.7237)},
        "MAE-small_ImageNet": {
            "IoU_FP": (0.4365, 0.4328),
            "IoU_SD": (0.5130, 0.5044), 
            "AUPR_FP": (0.6368, 0.6310), 
            "AUPR_SD": (0.7203, 0.7103)},
        "ResNet18_STED": {
            "IoU_FP": (0.3532, 0.4681), 
            "IoU_SD": (0.4594, 0.5170),
            "AUPR_FP": (0.5279, 0.6773), 
            "AUPR_SD": (0.6695, 0.7253)},
        "MAE-small_STED": {
            "IoU_FP": (0.4578, 0.4271), 
            "IoU_SD": (0.5233, 0.5023), 
            "AUPR_FP": (0.6702, 0.6357), 
            "AUPR_SD": (0.7290, 0.7060)}
    },
    "Lioness": {
        "ResNet18_from-scratch": {
            "IoU_Cell": 0.9250,
            "IoU_Boundary": 0.3658,
            "AUPR_Cell": 0.9880,
            "AUPR_Boundary": 0.4881},
        "MAE-small_from-scratch": {
            "IoU_Cell": 0.9234,
            "IoU_Boundary": 0.3579,
            "AUPR_Cell": 0.9874,
            "AUPR_Boundary": 0.4879},
        "ResNet18_HPA": {
            "IoU_Cell": (0.9231, 0.9233), 
            "IoU_Boundary": (0.3638, 0.3717), 
            "AUPR_Cell": (0.9882, 0.9884), 
            "AUPR_Boundary": (0.4874, 0.5011)},
        "ResNet18_ImageNet": {
            "IoU_Cell": (0.9237, 0.9239), 
            "IoU_Boundary": (0.3640, 0.3708), 
            "AUPR_Cell": (0.9875, 0.9878), 
            "AUPR_Boundary": (0.4867, 0.4996)},
        "ResNet18_STED": {
            "IoU_Cell": (0.9248, 0.9194), 
            "IoU_Boundary": (0.3589, 0.3739), 
            "AUPR_Cell": (0.9882, 0.9876), 
            "AUPR_Boundary": (0.4848, 0.4981)}
    },
    "synaptic-semantic": {
        "MAE-small_from-scratch": {
            "IoU_Round": 0.0883,
            "IoU_Elongated": 0.0213,
            "IoU_Perforated": 0,
            "IoU_Multidomains": 0.0280,
            "AUPR_Round": 0.2459,
            "AUPR_Elongated": 0.3078,
            "AUPR_Perforated": 0.2384,
            "AUPR_Multidomains": 0.1906},
        "MAE-small_HPA": {
            "IoU_Round": (0.1546, 0.0932),
            "IoU_Elongated": (0.1773, 0.0259),
            "IoU_Perforated": (0, 0),
            "IoU_Multidomains": (0.0641, 0.0381),
            "AUPR_Round": (0.3409, 0.2704),
            "AUPR_Elongated": (0.4763, 0.3904),
            "AUPR_Perforated": (0.2725, 0.2617),
            "AUPR_Multidomains": (0.2703, 0.2420)},
        "MAE-small_ImageNet": {
            "IoU_Round": (0.0834, 0.0978),
            "IoU_Elongated": (0.0725, 0.0343),
            "IoU_Perforated": (0, 0),
            "IoU_Multidomains": (0.0450, 0.0385),
            "AUPR_Round": (0.3071, 0.2785),
            "AUPR_Elongated": (0.3959, 0.3903),
            "AUPR_Perforated": (0.2381, 0.2702),
            "AUPR_Multidomains": (0.2703, 0.2440)},
        "MAE-small_STED": {
            "IoU_Round": (0.1469, 0.0938),
            "IoU_Elongated": (0.1863, 0.0964),
            "IoU_Perforated": (0, 0),
            "IoU_Multidomains": (0.0801, 0.0341),
            "AUPR_Round": (0.3509, 0.2875),
            "AUPR_Elongated": (0.4650, 0.4039),
            "AUPR_Perforated": (0.2556, 0.2240),
            "AUPR_Multidomains": (0.2969, 0.2494)}
    },
}