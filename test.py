import tensorflow as tf
import tensorflow_hub as hub

# Load the Gecko model
model = hub.load("@gecko/gecko-1b-en-tpu/3") # You can choose other models from go/gecko-models
serving_fn = model.signatures["serving_default"]
encoder = lambda x: serving_fn(x)["encodings"].numpy()

# Input text
query_texts = tf.constant(["This is a sample text.", "Another text example."])

# Get embeddings
query_embeds = encoder(query_texts)

# Now you can use query_embeds for your application (e.g., similarity, retrieval)
print(query_embeds)
