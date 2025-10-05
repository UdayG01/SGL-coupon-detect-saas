import onnx

# Load your ONNX model
model = onnx.load("model/best_fixed_2.onnx")

# Get input tensor info
for input_tensor in model.graph.input:
    shape = []
    for dim in input_tensor.type.tensor_type.shape.dim:
        if dim.dim_value > 0:
            shape.append(dim.dim_value)
        else:
            shape.append("dynamic")
    print(f"Input: {input_tensor.name}, Shape: {shape}")
