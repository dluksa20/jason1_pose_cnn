import tensorflow as tf
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import keras.backend as K
from tqdm import tqdm
from darknet import darknet19
from utils import HomoscedasticLoss


model = darknet19((640, 640, 3))
data = pd.read_csv('dataset_1.csv')
# data = data.head(898)

train_data, val_data = train_test_split(data, test_size=0.1, random_state=True)

train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.4,
    height_shift_range=0.4,
    zoom_range=0.4,
    horizontal_flip=True,
)

val_datagen = ImageDataGenerator(
    rescale=1.0/255.0)

batch_size = 6

train_generator = train_datagen.flow_from_dataframe(
    train_data, x_col='Image path', y_col=['t1', 't2', 't3', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32'],
    target_size=(640, 640), batch_size=batch_size,
    class_mode='raw',
    shuffle=True,
    seed=True
)

val_generator = val_datagen.flow_from_dataframe(
    val_data, x_col='Image path', y_col=['t1', 't2', 't3', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32'],
    target_size=(640, 640), batch_size=batch_size,
    class_mode='raw',
    shuffle=False
)

num_epochs = 50

loss = HomoscedasticLoss(0.0, -3.0)

opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.002)


for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    train_loss = 0
    train_steps = 0

    # Training loop
    with tqdm(total=len(train_generator), desc="Training", unit="batch") as pbar:
        for batch_x, batch_y in train_generator:
            with tf.GradientTape() as tape_sigma:
                predictions = model(batch_x, training=True)
                loss_value = loss(batch_y, predictions)
            gradients = tape_sigma.gradient(loss_value, model.trainable_variables+loss.trainable_variables)
            opt.apply_gradients(zip(gradients, model.trainable_variables+loss.trainable_variables))
    
            train_steps += 1
            # loss_value = K.mean(loss_value)
            train_loss += loss_value
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{loss_value.numpy():.4f}", "st": f"{loss.s_hat_t.numpy():.4f}", "sq": f"{loss.s_hat_q.numpy():.4f}"})
            if train_steps >= len(train_generator):
                break

            
    # Validation loop
    val_loss = 0
    val_steps = 0

    with tqdm(total=len(val_generator), desc="Validation", unit="batch") as pbar:
        for batch_x, batch_y in val_generator:
            predictions = model(batch_x, training=False)
            val_loss_value = loss(batch_y, predictions)

            val_steps += 1
            val_loss_value = K.mean(val_loss_value)
            val_loss += val_loss_value
            pbar.update(1)
            pbar.set_postfix({"Loss": f"{val_loss_value.numpy():.4f}", "st": f"{loss.s_hat_t.numpy():.4f}", "sq": f"{loss.s_hat_q.numpy():.4f}"})
            if val_steps >= len(val_generator):
                break

    # Calculate average losses for the epoch
    avg_train_loss = train_loss / train_steps
    avg_val_loss = val_loss / val_steps
    
    # Print epoch summary
    print(f"Epoch {epoch+1}/{num_epochs} - Avg Train Loss: {avg_train_loss:.4f} - Avg Val Loss: {avg_val_loss:.4f}")
    print()


# Save the trained model
model.save('trained_model.h5')
print('Training finished!')
            
