import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # Disable TensorFlow 2.x behavior
import tf2onnx

# Load the TensorFlow model
model = tf.keras.models.load_model('Crop Disease Prediction Model_CNN.h5')

# Compile the model (optional, but good practice)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define the input signature
spec = (tf.TensorSpec((None, 128, 128, 3), tf.float32, name="input"),)

# Convert the model to ONNX format
output_path = "model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

# Save the model to a file
with open(output_path, "wb") as f:
    f.write(model_proto.SerializeToString())
