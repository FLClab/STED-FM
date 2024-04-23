SUPERVISED = {
    "MAE" : {
        "synaptic-proteins": 0.742,
        "optim": 0.909,
    },
    "RESNET18": {
        "synaptic-proteins": None,
        "optim": None,
    },
    "RESNET50": {
        "synaptic-proteins": None,
        "optim": None,
    }
}

RESULTS = {
    "MAE": 
        {   
            "synaptic-proteins":
                {
                    "ImageNet": {
                        "KNN": 0.397,
                        "linear-probing": 0.499,
                        "fine-tuning": 0.720,
                    },
                    "CTC": {
                        "KNN": 0.409,
                        "linear-probing": 0.501,
                        "fine-tuning": 0.733,
                    },
                    "STED": {
                        "KNN": 0.749,
                        "linear-probing": 0.780,
                        "fine-tuning": 0.817,
                    }
                },
        "optim": 
            {
                "ImageNet": {
                        "KNN": 0.797,
                        "linear-probing": 0.881,
                        "fine-tuning": 0.881,
                    },
                    "CTC": {
                        "KNN": 0.860,
                        "linear-probing": 0.910,
                        "fine-tuning": 0.927,
                    },
                    "STED": {
                        "KNN": 0.954,
                        "linear-probing": 0.975,
                        "fine-tuning": 0.950,
                    }
            }
    },
    "RESNET18": {
        "synaptic-proteins": {
            "ImageNet": {
                "KNN": 0.679,
                "linear-probing": 0.664,
                "fine-tuning": 0.806,
            },
            "CTC": {
                "KNN": 0.533,
                "linear-probing": 0.665,
                "fine-tuning": 0.817,
            },
            "STED": {
                "KNN": 0.698,
                "linear-probing": 0.716,
                "fine-tuning": 0.843,
            }
        },
        "optim": {
             "ImageNet": {
                "KNN": 0.925,
                "linear-probing": 0.916,
                "fine-tuning": 0.982,
            },
            "CTC": {
                "KNN": 0.902,
                "linear-probing": 0.922,
                "fine-tuning": 0.947,
            },
            "STED": {
                "KNN": 0.948,
                "linear-probing": 0.966,
                "fine-tuning": 0.970,
            }

        }
    },
    "RESNET50": {
        "synaptic-proteins": {
            "ImageNet": {
                "KNN": 0.6674,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "CTC": {
                "KNN": 0.5434,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "STED": {
                "KNN": 0.6717,
                "linear-probing": None,
                "fine-tuning": None,
            }
        },
        "optim": {
             "ImageNet": {
                "KNN": 0.938,
                "linear-probing": 0.927,
                "fine-tuning": 0.954,
            },
            "CTC": {
                "KNN": 0.8750,
                "linear-probing": 0.927,
                "fine-tuning": 0.943,
            },
            "STED": {
                "KNN": 0.9566,
                "linear-probing": 0.902,
                "fine-tuning": 0.957,
            }

        }
    },
    "MICRANET": {
        "synaptic-proteins": {
            "ImageNet": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "CTC": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "STED": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            }
        },
        "optim": {
            "ImageNet": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "CTC": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "STED": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            }
        }
    },
    "CONVNEXT_SMALL": {
        "synaptic-proteins": {
            "ImageNet": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "CTC": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "STED": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            }
        },
        "optim": {
            "ImageNet": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "CTC": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            },
            "STED": {
                "KNN": None,
                "linear-probing": None,
                "fine-tuning": None,
            }
        }
    }
}