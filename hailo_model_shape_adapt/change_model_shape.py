import onnx
from onnx import shape_inference

def print_model_input_shape(model):
    for input_tensor in model.graph.input:
        shape = []
        for dim in input_tensor.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("dynamic")
        print(f"Input: {input_tensor.name}, Shape: {shape}")

# Load your original ONNX model
model_path = "model/best_5.onnx"
model = onnx.load(model_path)

print("Before fixing:")
print_model_input_shape(model)

# Fix the input shape to [1, 3, 640, 640]
input_tensor = model.graph.input[0]
input_tensor.type.tensor_type.shape.dim[0].dim_value = 1   # batch
input_tensor.type.tensor_type.shape.dim[1].dim_value = 3   # channels
input_tensor.type.tensor_type.shape.dim[2].dim_value = 640 # height
input_tensor.type.tensor_type.shape.dim[3].dim_value = 640 # width

# Run shape inference to propagate shapes through the graph
model = shape_inference.infer_shapes(model)

# Save the fixed model
fixed_model_path = "model/best_fixed_5.onnx"
onnx.save(model, fixed_model_path)

print("\nAfter fixing:")
print_model_input_shape(model)

print(f"\nFixed ONNX model saved to: {fixed_model_path}")
