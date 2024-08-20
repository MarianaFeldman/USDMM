import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks

def build_model(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Initial Convolutional layers with batch normalization
    x = layers.Conv2D(32, (3, 3), activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)

    # Residual block
    residual = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    residual = layers.BatchNormalization()(residual)
    residual = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(residual)
    residual = layers.BatchNormalization()(residual)

    # Add the residual connection
    x = layers.Add()([residual, x])
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.5)(x)

    # Flatten and fully connected layers
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)  # Output layer with sigmoid activation

    # Create the final model
    model = models.Model(inputs=inputs, outputs=outputs)

    return model

def test_cnn(train_images, train_labels, val_images, val_labels, test_images, test_labels, test_name):

    print(f'Running {test_name}')
    model = build_model((28, 28, 1))
    model.compile(optimizer=optimizers.AdamW(learning_rate=0.001),
                loss='mean_squared_error',  # Using MSE as loss function
                metrics=['accuracy'])

    # Data Augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1
    )

    # Learning Rate Scheduler
    lr_scheduler = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    # Train the model
    history = model.fit(datagen.flow(train_images, train_labels, batch_size=50),
                        validation_data=(val_images, val_labels),
                        epochs=100,
                        callbacks=[lr_scheduler, callbacks.EarlyStopping(monitor='val_loss', patience=100)])


    predictions_train = model.predict(train_images).flatten()
    predictions_val = model.predict(val_images)
    predictions_test = model.predict(test_images).flatten()

    error_prob_train = np.mean(np.abs(train_labels - predictions_train))
    error_prob_val = np.mean(np.abs(val_labels - predictions_val))
    error_prob_test = np.mean(np.abs(test_labels - predictions_test))

    predct_bin_train = (predictions_train > 0.5).astype('float32')
    predct_bin_val = (predictions_val > 0.5).astype('float32')
    predct_bin_test = (predictions_test > 0.5).astype('float32')

    error_train = np.mean((train_labels - predct_bin_train)**2)
    error_val = np.mean((val_labels - predct_bin_val)**2)
    error_test = np.mean((test_labels - predct_bin_test)**2)

    print(f'Error train class: {error_train:.4f} / Error train prob: {error_prob_train:.4f}')
    print(f'Error val class: {error_val:.4f} / Error val prob: {error_prob_val:.4f}')
    print(f'Error test class: {error_test:.4f} / Error test prob: {error_prob_test:.4f}')
    print('----------------------------------------------------------')

def transform_mnist_digit(train_size, val_size, train_images, train_labels):
  np.random.seed(0)

  train_dig_idx = np.where(train_labels == 1)[0]

  train_non_dig_idx = np.where(train_labels == 0)[0]

  num_train_zeros = (train_size+val_size)//3

  train_selected_zeros_idx = np.random.choice(train_dig_idx, num_train_zeros, replace=False)
  num_train_non_zeros = (train_size+val_size) - num_train_zeros
  train_selected_non_zeros_idx = np.random.choice(train_non_dig_idx, num_train_non_zeros, replace=False)

  train_idx = np.concatenate([train_selected_zeros_idx, train_selected_non_zeros_idx])
  np.random.shuffle(train_idx)

  x_train = np.array(train_images[train_idx][:train_size])
  y_train = np.array(train_labels[train_idx][:train_size])

  x_val = np.array(train_images[train_idx][train_size:(train_size+val_size)])
  y_val = np.array(train_labels[train_idx][train_size:(train_size+val_size)])

  return x_train, y_train, x_val, y_val

def main():
   (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
   train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
   test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255

   train_labels = (train_labels == 1).astype('float32')
   test_labels = (test_labels == 1).astype('float32')
   
   for N in [200, 600]:
      train_images, train_labels, val_images, val_labels = transform_mnist_digit(1, N, 100, train_images, train_labels)
      test_name = f'Teste com {N} imagens de Treino'
      test_cnn(train_images, train_labels, val_images, val_labels, test_images, test_labels, test_name)


if __name__ == "__main__":
    main()

