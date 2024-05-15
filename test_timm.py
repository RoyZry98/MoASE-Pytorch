import timm
model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
print(model)