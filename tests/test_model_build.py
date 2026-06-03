
import torch 

from stedfm import get_pretrained_model_v2 

def load_stedfm_model():
    model, cfg = get_pretrained_model_v2(
        name="mae-lightning-small",
        weights="MAE_SMALL_STED",
        in_channels=1,
    ) 
    return model, cfg

def load_mcms_stedfm_model():
    model, cfg = get_pretrained_model_v2(
        name="mae-mcms-lightning-small",
        weights="MAE_MCMS_SMALL_STED",
        in_channels=1,
    ) 
    return model, cfg

if __name__ == "__main__":
    stedfm_model, _ = load_stedfm_model()
    mcms_model, _ = load_mcms_stedfm_model()

    # Compare the state dicts of the ViT backbones of both models to ensure they are identical, as they were both pretrained with the same weights on the same dataset
    stedfm_state_dict = stedfm_model.backbone.vit.state_dict()
    mcms_state_dict = mcms_model.backbone.vit.state_dict()

    flag = False
    for key in stedfm_state_dict.keys():
        if key in mcms_state_dict:
            if not torch.equal(stedfm_state_dict[key], mcms_state_dict[key]):
                flag = True
                average_diff = torch.mean(torch.abs(stedfm_state_dict[key] - mcms_state_dict[key])).item()
                print(f"Mismatch in key {key}, average absolute difference: {average_diff}")
        else:
            print(f"Key {key} not found in MCMS model state dict")
    
    if not flag:
        print("The ViT backbones of both models are identical, as expected since they were pretrained with the same weights on the same dataset.")
    else:
        print("The ViT backbones of both models are not identical, which is unexpected since they were pretrained with the same weights on the same dataset. This could be due to differences in the model architecture or training process.")