import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

def create_resnet_model(num_classes=4):
    """
    Create a ResNet50-based model for skin feature classification
    """
    # Load pre-trained ResNet50 model
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model layers
    base_model.trainable = False
    
    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # Create the final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    return model

def train_model():
    """
    Train the ResNet50 model on the skin feature dataset
    """
    # Set up data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2,
        validation_split=0.2
    )
    
    # Check if dataset exists
    dataset_path = 'dataset'
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found. Please download and place the dataset folder in the project directory.")
        return None
    
    # Get class names
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"✅ Found classes: {class_names}")
    
    # Create data generators
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='training'
    )
    
    validation_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        subset='validation'
    )
    
    # Create model
    model = create_resnet_model(num_classes=len(class_names))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Train model
    print("🚀 Starting training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save the trained model
    model.save('dermalscan_resnet_finetuned_model.h5')
    print("✅ Model saved as 'dermalscan_resnet_finetuned_model.h5'")
    
    # Fine-tuning: Unfreeze some layers and train with lower learning rate
    print("🔧 Starting fine-tuning...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze layers except the last few
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    history_finetune = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        verbose=1
    )
    
    # Save the fine-tuned model
    model.save('dermalscan_resnet_model.h5')
    print("✅ Fine-tuned model saved as 'dermalscan_resnet_model.h5'")
    
    return model

if __name__ == '__main__':
    print("🎯 DermalScan AI - ResNet50 Model Training")
    print("=" * 50)
    
    # Check GPU availability
    if tf.config.list_physical_devices('GPU'):
        print("✅ GPU detected - training will be accelerated")
    else:
        print("⚠️ No GPU detected - training will use CPU (slower)")
    
    # Train the model
    model = train_model()
    
    if model:
        print("🎉 Training completed successfully!")
        print("📁 Model files saved:")
        print("   - dermalscan_resnet_model.h5 (fine-tuned)")
        print("   - dermalscan_resnet_finetuned_model.h5 (initial)")
    else:
        print("❌ Training failed. Please check the dataset and try again.")
