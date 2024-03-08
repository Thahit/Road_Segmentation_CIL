from Models.model3att import get_model

model = get_model()
pytorch_total_params = sum(p.numel() for p in model.parameters())
print(pytorch_total_params)
