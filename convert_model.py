import tensorflow as tf

# Load your existing model
model = tf.keras.models.load_model("model.keras", compile=False)

# Save as .h5
model.save("model.h5")

print("Model successfully converted to model.h5")