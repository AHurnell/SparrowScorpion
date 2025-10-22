import tensorflow as tf
import os

# --- Step 1: Load your existing .h5 model ---
print("📦 Loading model...")
model = tf.keras.models.load_model("drone_classifier.h5")
print("✅ Model loaded successfully!")

# --- Step 2: Export it to TensorFlow SavedModel format ---
print("💾 Exporting to TensorFlow SavedModel format...")
export_dir = "saved_model_export"
os.makedirs(export_dir, exist_ok=True)
model.export(export_dir)  # ✅ new API for TF 2.16+
print("✅ SavedModel exported!")

# --- Step 3: Convert to TensorFlow Lite ---
print("⚙️ Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)

# Optional optimizations for smaller + faster model
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter._experimental_lower_tensor_list_ops = False  # fixes MobileNetV2 variable issues

try:
    tflite_model = converter.convert()
    with open("drone_classifier.tflite", "wb") as f:
        f.write(tflite_model)
    print("🎉 Conversion successful! File saved as drone_classifier.tflite")
except Exception as e:
    print("❌ Conversion failed:", e)
