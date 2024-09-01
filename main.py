
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import legacy
from utils import PoseLosses, PrintSPSQ
from keras.callbacks import ReduceLROnPlateau,  EarlyStopping, ModelCheckpoint
from models import darknet19, EfficientNet, build_efficientnet_b0_scaled


if __name__ == '__main__':

    # create model 
    model = darknet19((300, 300, 3))
    batch_size = 12
    
    # import dataset
    data = pd.read_csv('dataset_LSTM.csv')
    # dataval = pd.read_csv('dataset360.csv')
    # split the data in validation and test sets
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=True)  # Set a fixed random state

    # data augmentation
    datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=40,
        width_shift_range=0.4,
        height_shift_range=0.4,
        zoom_range=0.4,
        horizontal_flip=True
    )
    
    # training data
    train_generator = datagen.flow_from_dataframe(
        train_data, x_col='Image path', y_col=['t1', 't2', 't3', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32'],
        target_size=(300, 300), batch_size=batch_size,
        class_mode='raw',
        shuffle=True
    )
    # Validation data
    val_generator = datagen.flow_from_dataframe(
        val_data, x_col='Image path', y_col=['t1', 't2', 't3', 'r11', 'r21', 'r31', 'r12', 'r22', 'r32'],
        target_size=(300, 300), batch_size=batch_size,
        class_mode='raw',
        shuffle=False
    )

    # learning rate reducer callback based on validation loss
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=4,
        min_lr=1e-5,
        verbose=1
    )
    # early stop callback if model converged and validation loss is stable
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True
    )
    # Create checkpoint call back to save the best model with lowest validation loss
    checkpoint = ModelCheckpoint(filepath='best_model.h5',  
                             monitor='val_loss',        
                             save_best_only=True,       
                             mode='min',                 
                             verbose=1)   
    

    # create loss function instance and assign initial translation and rotation weights (the numerical values enterd into layer directly)
    # loss = PoseLosses.Adaptive_loss(model.get_layer('trainable_spsq_layer').SP, model.get_layer('trainable_spsq_layer').SQ)
    loss = PoseLosses.Adaptive_loss(0.5,-3.0)

    # initialize optimizer and assign initial learning rate
    opt = legacy.Adam(learning_rate=0.002)
    # Compile the model
    model.compile(optimizer=opt, loss=loss, metrics=['mean_absolute_error'])
    # Create call back to print to terminal updated los weights SP and SQ
    print_sp_sq_callback = PrintSPSQ()

    # training
    model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=50,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[print_sp_sq_callback, lr_reducer, early_stopping, checkpoint]
    )
    # Save the trained model with a specific path
    model.save('trained_model.h5')
    print('Training finished!')
