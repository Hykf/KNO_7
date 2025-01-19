import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import kagglehub


data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2)
])

if __name__ == '__main__':
    print("start")
    data_dir = kagglehub.dataset_download("khushikhushikhushi/dog-breed-image-dataset")
    data_dir += "\\dataset"

    batch_size = 32
    img_height, img_width = 128, 128

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels=None,
        validation_split=0.3,
        subset="training",
        seed=707,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    train_ds = train_ds.map(lambda x: (x / 255.0, x / 255.0))
    latent_dim = 2

    encoder = models.Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        data_augmentation,
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(latent_dim)
    ])

    decoder = models.Sequential([
        layers.Input(shape=(latent_dim,)),
        layers.Dense(16 * 16 * 128, activation='relu'),
        layers.Reshape((16, 16, 128)),
        layers.Conv2DTranspose(128, 3, activation='relu', padding='same'),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(64, 3, activation='relu', padding='same'),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(32, 3, activation='relu', padding='same'),
        layers.UpSampling2D(),
        layers.Conv2DTranspose(3, 3, activation='sigmoid', padding='same')
    ])

    autoencoder = models.Model(encoder.inputs, decoder(encoder.outputs))
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    epochs = 100
    history = autoencoder.fit(train_ds, epochs=epochs)
    random_latent_vector = np.random.normal(size=(1, latent_dim))
    generated_image = decoder.predict(random_latent_vector)
    plt.imshow(generated_image[0])
    plt.axis("off")
    plt.show()

    for batch in train_ds.take(1):
        original_images = batch[0].numpy()
        break

    encoded_imgs = encoder.predict(original_images)
    decoded_imgs = autoencoder.predict(original_images)

    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original_images[i])
        plt.axis("off")

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i])
        plt.axis("off")

    plt.show()