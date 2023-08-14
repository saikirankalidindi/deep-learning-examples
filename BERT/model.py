import tensorflow as tf
from transformers import BertTokenizer, BertModel

# Load BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

# Example text
text = "This is an example sentence with words."

# Tokenize the text and get BERT embeddings
inputs = tokenizer(text, return_tensors='tf')
outputs = bert_model(**inputs)

# Get the pooled output from BERT (CLS token representation)
pooled_output = outputs.pooler_output

# Define the LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(pooled_output.shape[1], pooled_output.shape[2])),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Generate example data (you will replace this with your own dataset)
X_train = tf.random.normal((100, pooled_output.shape[1], pooled_output.shape[2]))
y_train = tf.random.uniform((100, 1), 0, 2, dtype=tf.int32)

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32)
