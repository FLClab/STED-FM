import timm
from timm.models.vision_transformer import vit_small_patch16_224

if __name__=="__main__":
    vit = vit_small_patch16_224(in_chans=3, pretrained=True)
