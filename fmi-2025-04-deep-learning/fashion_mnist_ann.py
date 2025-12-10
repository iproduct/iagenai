import  matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras.datasets import fashion_mnist, mnist


def create_ann():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

def train_ann(model, x_train, y_train, x_test, y_test, epochs = 5):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x = x_train, y = y_train, epochs = epochs, validation_data = (x_test, y_test))

def test_ann(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print('Test accuracy:', test_acc)
    print('Test loss:', test_loss)

if __name__ == "__main__":
    print(keras.__version__)
    # (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.ndim, x_train.shape, x_train.dtype)
    x_slice = x_train[:20, :, :]
    y_slice = y_train[:20]

    for i in range(20):
        ex = x_slice[i]
        img = plt.imshow(ex, cmap=plt.cm.binary)
        plt.title(y_slice[i])
        plt.show()

    model = create_ann()
    model.summary()

    train_ann(model, x_train, y_train, x_test, y_test, 10)
    test_ann(model, x_test, y_test)




