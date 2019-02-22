import torch
import torch.nn as nn
import torchvision

# An instance of your model.
model = torchvision.models.resnet18()
model.fc = nn.Linear(in_features=512, out_features=120, bias=True)
model.load_state_dict(torch.load('/home/cbc/disk2/dog_classification/dog_recognition_resnet18-100-32-75.00-0.001.pkl'))
model.eval()
# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 224, 224)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)

# save
traced_script_module.save("model.pt")
