# (Frozen, Pretrained)

RESULTS = {
    "F-Actin": {
        "ResNet18_from-scratch": {
            "IoU_Rings": 0.4054,
            "IoU_Fibers": 0.5190,
            "AUPR_Rings": 0.6076,
            "AUPR_Fibers": 0.7370},
        "MAE-small_from-scratch": {
            "IoU_Rings": 0.4189,
            "IoU_Fibers": 0.6198,
            "AUPR_Rings": 0.5315,
            "AUPR_Fibers": 0.7248},
        "ResNet18_HPA": {
            "IoU_Rings": (0.3596, 0.3756), 
            "IoU_Fibers": (0.5655, 0.6143), 
            "AUPR_Rings": (0.6031, 0.6138), 
            "AUPR_Fibers": (0.7685, 0.7731)},
        "MAE-small_HPA": {
            "IoU_Rings": (0.4431, 0.4450), 
            "IoU_Fibers": (0.6228, 0.5974), 
            "AUPR_Rings": (0.5831, 0.5956), 
            "AUPR_Fibers": (0.7442, 0.7440)},
        "ResNet18_ImageNet": {
            "IoU_Rings": (0.3360, 0.3324), 
            "IoU_Fibers": (0.5562, 0.5984), 
            "AUPR_Rings": (0.6075, 0.6141), 
            "AUPR_Fibers": (0.7432, 0.7613)},
        "MAE-small_ImageNet": {
            "IoU_Rings": (0.0260, 0.1557), 
            "IoU_Fibers": (0.1798, 0.2288), 
            "AUPR_Rings": (0.5481, 0.5749), 
            "AUPR_Fibers": (0.7299, 0.7243)},
        "ResNet18_STED": {
            "IoU_Rings": (0.3438, 0.3096), 
            "IoU_Fibers": (0.4917, 0.5649), 
            "AUPR_Rings": (0.5699, 0.5889),
             "AUPR_Fibers": (0.7448, 0.7544)},
        "MAE-small_STED": {
            "IoU_Rings": (0.4257, 0.4286), 
            "IoU_Fibers": (0.6263, 0.6068), 
            "AUPR_Rings": (0.6037, 0.6214),
            "AUPR_Fibers": (0.7622, 0.7733)},
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
            "IoU_FP": 0.4669,
            "IoU_SD": 0.5275,
            "AUPR_FP": 0.6841,
            "AUPR_SD": 0.7245},
        "MAE-small_from-scratch": {
            "IoU_FP": 0.3333,
            "IoU_SD": 0.4539,
            "AUPR_FP": 0.5303,
            "AUPR_SD": 0.6435},
        "ResNet18_HPA": {
            "IoU_FP": (0.4294, 0.4857), 
            "IoU_SD": (0.5191, 0.5378), 
            "AUPR_FP": (0.6456, 0.6995), 
            "AUPR_SD": (0.7203, 0.7360)},
        "MAE-small_HPA": {
            "IoU_FP": (0.4302, 0.4521), 
            "IoU_SD": (0.5090, 0.5252), 
            "AUPR_FP": (0.6387, 0.6562), 
            "AUPR_SD": (0.7046, 0.7211)},
        "ResNet18_ImageNet": {
            "IoU_FP": (0.4192, 0.4785), 
            "IoU_SD": (0.5061, 0.5328), 
            "AUPR_FP": (0.6301, 0.6956), 
            "AUPR_SD": (0.7093, 0.7311)},
        "MAE-small_ImageNet": {
            "IoU_FP": (0.3607, 0.4346),
            "IoU_SD": (0.4525, 0.5053), 
            "AUPR_FP": (0.5427, 0.6416), 
            "AUPR_SD": (0.6543, 0.6955)},
        "ResNet18_STED": {
            "IoU_FP": (0.3823, 0.4440), 
            "IoU_SD": (0.4852, 0.5247),
            "AUPR_FP": (0.5735, 0.6604), 
            "AUPR_SD": (0.6898, 0.7250)},
        "MAE-small_STED": {
            "IoU_FP": (0.4300, 0.4508), 
            "IoU_SD": (0.5238, 0.5291), 
            "AUPR_FP": (0.6419, 0.6635), 
            "AUPR_SD": (0.7219, 0.7273)}
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