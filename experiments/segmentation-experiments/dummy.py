from timm.models.vision_transformer import vit_small_patch16_224, vit_tiny_patch16_224, vit_base_patch16_224, vit_large_patch16_224, VisionTransformer


if __name__=="__main__":
    vit = vit_small_patch16_224(in_chans=1, pretrained=False)
    print(vit)
