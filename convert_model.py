import tensorflow as tf

# Load original model
model = tf.keras.models.load_model("model.keras", compile=False)

# 🔥 Re-save in legacy-compatible format
tf.keras.models.save_model(
    model,
    "model_fixed.h5",
    save_format="h5"
)

print("Saved as model_fixed.h5")