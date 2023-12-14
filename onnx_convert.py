# trtexec --workspace=10240 --verbose --onnx=MSRA-TD500.onnx --saveEngine=MSRA-TD500.trt

import torch
import numpy as np
import onnxruntime as rt

from network.textnet import TextNet

model = TextNet()
model_name = "MSRA-TD500"
# model_name = "TotalText"
model_path = f"pretrained/{model_name}.pth"
state = torch.load(model_path, map_location=torch.device("cpu"))
model.load_state_dict(state["model"])
model.eval()

size = 1024
x = torch.randn(1, 3, size, size)
torch.onnx.export(
    model,
    x,
    f"pretrained/{model_name}.onnx",
    input_names=["input"],
    output_names=["output"],
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    verbose=False,
)

x = torch.randn(1, 3, size, size)
y = model(x).detach().cpu().numpy()
print(y.shape)

sess = rt.InferenceSession(f"pretrained/{model_name}.onnx")
input_name = sess.get_inputs()[0].name
z = sess.run(None, {input_name: x.numpy()})[0]

print(z.shape)
print(np.abs(y - z).max())
