# /// script
# requires-python = ">=3.10, <3.14"
# dependencies = [
#   "tensorflow",
#   "transformers[tf]<5.0",
#   "safetensors<0.4.0",
#   "Pillow",
#   "iree-compiler",
#   "iree-tools-tf",
# ]
# ///

import tensorflow as tf
from transformers import AutoImageProcessor, TFResNetForImageClassification
import sys
import shutil
from PIL import Image
import os
from iree.compiler import tf as tfc

ARTIFACTS_DIR = "../build/mlir_artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Download model and preprocessor from HuggingFace
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = TFResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Load image and process it
file_name = sys.argv[1]
image = Image.open(file_name).convert("RGB")
processed_image = processor(image, return_tensors="np")

# save raw bytes to file
bin_name = os.path.splitext(os.path.basename(file_name))[0] + '.bin'
bin_path = os.path.join(ARTIFACTS_DIR, bin_name)
with open(bin_path, "wb") as f:
    f.write(processed_image['pixel_values'].tobytes())

saved_model_dir = os.path.join(ARTIFACTS_DIR, "resnet50")
shutil.rmtree(saved_model_dir, ignore_errors=True)

# fix model input shape to 1, 3, 224, 224
def model_exporter(model: tf.keras.Model):
    m_call = tf.function(model.call).get_concrete_function(
        tf.TensorSpec(
            shape=[None, 3, 224, 224], dtype=tf.float32, name='pixel_values'
        )
    )
    
    @tf.function(input_signature=[tf.TensorSpec([1, 3, 224, 224], tf.float32)])
    def serving_fn(input):
        return model(**processed_image).logits

    return serving_fn

model.save_pretrained(saved_model_dir, saved_model=True, signatures={'serving_default': model_exporter(model)})

# save id2label
with open(os.path.join(ARTIFACTS_DIR, "id2label.txt"), "w") as f:
    for i in range(len(model.config.id2label)):
        f.write(model.config.id2label[i] + '\n')

os.makedirs("../compiled_models", exist_ok=True)
mlir_bytes = tfc.compile_saved_model(
    os.path.join(saved_model_dir, "saved_model", "1"),
    import_only=True,
    exported_names=["serving_default"],
)
with open("../compiled_models/resnet50.mlir", "wb") as f:
    f.write(mlir_bytes)