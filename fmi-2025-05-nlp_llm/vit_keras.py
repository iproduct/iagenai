import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ----------------------------
# Load CIFAR-10
# ----------------------------
(x_train, _), _ = keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0

# ----------------------------
# Data augmentation (SimCLR)
# ----------------------------
augment = keras.Sequential([
    layers.Resizing(224, 224),
    layers.RandomCrop(224, 224),
    layers.RandomFlip("horizontal"),
    layers.RandomContrast(0.4),
])

def make_views(x):
    return augment(x), augment(x)

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(5000)
    .batch(128)
    .map(make_views, num_parallel_calls=tf.data.AUTOTUNE)
    .prefetch(tf.data.AUTOTUNE)
)

# ----------------------------
# ViT Components
# ----------------------------
class Patches(layers.Layer):
    def call(self, x):
        patches = tf.image.extract_patches(
            x, [1,16,16,1], [1,16,16,1], [1,1,1,1], "VALID"
        )
        return tf.reshape(patches, [tf.shape(x)[0], -1, patches.shape[-1]])

class PatchEncoder(layers.Layer):
    def __init__(self):
        super().__init__()
        self.proj = layers.Dense(256)
        self.pos = layers.Embedding(196, 256)

    def call(self, x):
        pos = tf.range(tf.shape(x)[1])
        return self.proj(x) + self.pos(pos)

def transformer(x):
    attn = layers.MultiHeadAttention(4, 256)(x, x)
    x = layers.LayerNormalization()(x + attn)
    mlp = layers.Dense(1024, activation="gelu")(x)
    mlp = layers.Dense(256)(mlp)
    return layers.LayerNormalization()(x + mlp)

def build_vit():
    inp = layers.Input((224, 224, 3))
    x = Patches()(inp)
    x = PatchEncoder()(x)
    for _ in range(6):
        x = transformer(x)
    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(inp, x)

# ----------------------------
# Contrastive Model
# ----------------------------
def contrastive_loss(z1, z2, t=0.2):
    z1 = tf.math.l2_normalize(z1, 1)
    z2 = tf.math.l2_normalize(z2, 1)
    logits = tf.matmul(z1, z2, transpose_b=True) / t
    labels = tf.range(tf.shape(z1)[0])
    return tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True) +
        tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    )

class ContrastiveModel(keras.Model):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.projector = keras.Sequential([
            layers.Dense(256, activation="relu"),
            layers.Dense(256),
        ])

    def train_step(self, data):
        x1, x2 = data
        with tf.GradientTape() as tape:
            z1 = self.projector(self.encoder(x1, training=True))
            z2 = self.projector(self.encoder(x2, training=True))
            loss = contrastive_loss(z1, z2)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"loss": loss}

# ----------------------------
# Train
# ----------------------------
encoder = build_vit()
model = ContrastiveModel(encoder)
model.compile(optimizer=keras.optimizers.Adam(3e-4))
model.fit(dataset, epochs=10)
