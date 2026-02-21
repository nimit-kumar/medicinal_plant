import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2, ResNet50V2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATASET_PATH = r"C:\Users\nimit\Music\.vscode\medicinal plant\Medicinal-plant-dataset"
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 15
NUM_CLASSES = 40

# Clear any previous models from memory
keras.backend.clear_session()

# Get all class names
class_names = sorted([d for d in os.listdir(DATASET_PATH) 
                     if os.path.isdir(os.path.join(DATASET_PATH, d))])
print(f"Found {len(class_names)} classes: {class_names}")

# Check class distribution
class_image_counts = {}
for class_name in class_names:
    class_path = os.path.join(DATASET_PATH, class_name)
    num_images = len([f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
    class_image_counts[class_name] = num_images
    print(f"{class_name}: {num_images} images")

def calculate_class_weights(train_generator):
    """Calculate class weights for imbalanced dataset"""
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weight_dict = dict(enumerate(class_weights))
    return class_weight_dict

def create_data_generators():
    """Create data generators with proper preprocessing"""
    
    # Training data with augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=0.2
    )
    
    # Validation/Test data with only rescaling
    test_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create generators
    train_generator = train_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    validation_generator = test_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    # Create a separate test generator
    test_generator = test_datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    return train_generator, validation_generator, test_generator

def create_model_resnet():
    """Create model using ResNet50V2 as base (alternative to EfficientNet)"""
    
    # Load pre-trained ResNet50V2
    base_model = ResNet50V2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create new model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data augmentation (applied only during training)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.2)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Batch normalization
    x = layers.BatchNormalization()(x)
    
    # Dense layers with dropout
    x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def create_model_mobilenet():
    """Create model using MobileNetV2 as base"""
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Create new model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    
    # Data augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.RandomContrast(0.2)(x)
    
    # Base model
    x = base_model(x, training=False)
    
    # Global pooling
    x = layers.GlobalAveragePooling2D()(x)
    
    # Batch normalization
    x = layers.BatchNormalization()(x)
    
    # Dense layers with dropout
    x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer
    outputs = layers.Dense(NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    
    return model, base_model

def plot_training_history(history, save_path='training_history.png'):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_confusion_matrix_results(model, test_generator, class_names, history, timestamp):
    """Save confusion matrix and detailed results"""
    
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Get classification report
    report = classification_report(y_true, y_pred, 
                                 target_names=class_names, 
                                 output_dict=True)
    
    # Save confusion matrix as CSV
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f'confusion_matrix_{timestamp}.csv')
    
    # Save normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm_df = pd.DataFrame(cm_normalized, index=class_names, columns=class_names)
    cm_norm_df.to_csv(f'confusion_matrix_normalized_{timestamp}.csv')
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f'classification_report_{timestamp}.csv')
    
    # Plot and save confusion matrix
    plt.figure(figsize=(30, 25))
    
    # Plot raw counts
    plt.subplot(1, 2, 1)
    sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', 
                xticklabels=True, yticklabels=True, annot_kws={'size': 8})
    plt.title('Confusion Matrix (Counts)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    # Plot normalized
    plt.subplot(1, 2, 2)
    sns.heatmap(cm_norm_df, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=True, yticklabels=True, annot_kws={'size': 8})
    plt.title('Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_complete_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Calculate per-class accuracy
    class_accuracy = {}
    for i, class_name in enumerate(class_names):
        correct = cm[i, i]
        total = np.sum(cm[i, :])
        accuracy = correct / total if total > 0 else 0
        class_accuracy[class_name] = accuracy
    
    # Save class-wise accuracy
    class_acc_df = pd.DataFrame(list(class_accuracy.items()), 
                               columns=['Class', 'Accuracy'])
    class_acc_df = class_acc_df.sort_values('Accuracy', ascending=False)
    class_acc_df.to_csv(f'class_wise_accuracy_{timestamp}.csv', index=False)
    
    # Plot class-wise accuracy
    plt.figure(figsize=(15, 8))
    colors = ['green' if acc > 0.8 else 'orange' if acc > 0.6 else 'red' 
              for acc in class_acc_df['Accuracy']]
    plt.barh(class_acc_df['Class'], class_acc_df['Accuracy'], color=colors)
    plt.xlabel('Accuracy')
    plt.title('Class-wise Accuracy', fontsize=14, fontweight='bold')
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'class_wise_accuracy_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save training history
    with open(f'training_history_{timestamp}.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    # Save final metrics
    final_metrics = {
        'final_train_accuracy': history.history['accuracy'][-1],
        'final_val_accuracy': history.history['val_accuracy'][-1],
        'final_train_loss': history.history['loss'][-1],
        'final_val_loss': history.history['val_loss'][-1],
        'best_val_accuracy': max(history.history['val_accuracy']),
        'best_epoch': np.argmax(history.history['val_accuracy']) + 1,
        'test_accuracy': model.evaluate(test_generator, verbose=0)[1],
        'test_loss': model.evaluate(test_generator, verbose=0)[0]
    }
    
    # Add worst performing classes
    worst_classes = class_acc_df.tail(5)
    final_metrics['worst_performing_classes'] = worst_classes.to_dict('records')
    
    # Add best performing classes
    best_classes = class_acc_df.head(5)
    final_metrics['best_performing_classes'] = best_classes.to_dict('records')
    
    # Save metrics
    with open(f'final_metrics_{timestamp}.txt', 'w') as f:
        for key, value in final_metrics.items():
            if key not in ['worst_performing_classes', 'best_performing_classes']:
                f.write(f'{key}: {value}\n')
            else:
                f.write(f'\n{key}:\n')
                for item in value:
                    f.write(f"  {item['Class']}: {item['Accuracy']:.4f}\n")
    
    print(f"\nResults saved with timestamp: {timestamp}")
    print(f"Final Validation Accuracy: {final_metrics['final_val_accuracy']:.4f}")
    print(f"Best Validation Accuracy: {final_metrics['best_val_accuracy']:.4f} (Epoch {final_metrics['best_epoch']})")
    print(f"Test Accuracy: {final_metrics['test_accuracy']:.4f}")
    
    return final_metrics

def fine_tune_model(model, base_model, train_generator, validation_generator, class_weights, timestamp):
    """Fine-tune the model with gradual unfreezing"""
    
    # Unfreeze the base model
    base_model.trainable = True
    
    # Freeze early layers, train later layers
    # For MobileNetV2, freeze first 100 layers
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile with very low learning rate
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks for fine-tuning
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1),
        ModelCheckpoint(f'best_finetuned_model_{timestamp}.h5', 
                       monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Continue training
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=FINE_TUNE_EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    return history

def main():
    print("="*50)
    print("Medicinal Plant Classification Training")
    print("="*50)
    
    # Create timestamp
    start_time = datetime.now()
    timestamp = start_time.strftime("%Y%m%d_%H%M%S")
    print(f"Training started at: {start_time}")
    
    # Create data generators
    print("\n1. Creating data generators...")
    train_generator, validation_generator, test_generator = create_data_generators()
    
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {validation_generator.samples}")
    print(f"Test samples: {test_generator.samples}")
    
    # Calculate class weights
    print("\n2. Calculating class weights for imbalanced data...")
    class_weights = calculate_class_weights(train_generator)
    print("Class weights calculated")
    
    # Try different models if one fails
    print("\n3. Creating model...")
    try:
        # Try MobileNetV2 first (lighter and more compatible)
        print("Attempting to create MobileNetV2 model...")
        model, base_model = create_model_mobilenet()
        print("MobileNetV2 model created successfully!")
    except Exception as e:
        print(f"MobileNetV2 failed: {e}")
        print("Trying ResNet50V2...")
        try:
            model, base_model = create_model_resnet()
            print("ResNet50V2 model created successfully!")
        except Exception as e:
            print(f"ResNet50V2 failed: {e}")
            print("Creating custom CNN model as fallback...")
            # Create a simple custom CNN if transfer learning fails
            model = create_custom_cnn()
            base_model = None
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    model.summary()
    
    # Callbacks for initial training
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1),
        ModelCheckpoint(f'best_initial_model_{timestamp}.h5', 
                       monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Train initial model
    print("\n4. Training initial model...")
    initial_history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Fine-tune if using transfer learning
    if base_model is not None:
        print("\n5. Fine-tuning model...")
        fine_tune_history = fine_tune_model(model, base_model, train_generator, 
                                           validation_generator, class_weights, timestamp)
        
        # Combine histories
        combined_history = type('obj', (object,), {'history': {}})
        for key in initial_history.history.keys():
            combined_history.history[key] = (
                initial_history.history[key] + 
                fine_tune_history.history.get(key, [])
            )
    else:
        combined_history = initial_history
    
    # Plot training history
    print("\n6. Plotting training history...")
    plot_training_history(combined_history, f'training_history_{timestamp}.png')
    
    # Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save confusion matrix and results
    print("\n8. Saving confusion matrix and detailed results...")
    final_metrics = save_confusion_matrix_results(
        model, test_generator, class_names, combined_history, timestamp
    )
    
    # Save the final model
    model.save(f'medicinal_plant_model_{timestamp}.h5')
    print(f"\nModel saved as: medicinal_plant_model_{timestamp}.h5")
    
    # Save class indices
    class_indices = train_generator.class_indices
    with open(f'class_indices_{timestamp}.pkl', 'wb') as f:
        pickle.dump(class_indices, f)
    
    # Save configuration
    config = {
        'dataset_path': DATASET_PATH,
        'img_size': IMG_SIZE,
        'batch_size': BATCH_SIZE,
        'initial_epochs': EPOCHS,
        'fine_tune_epochs': FINE_TUNE_EPOCHS,
        'num_classes': NUM_CLASSES,
        'class_names': class_names,
        'class_image_counts': class_image_counts,
        'timestamp': timestamp,
        'training_time': str(datetime.now() - start_time)
    }
    
    with open(f'training_config_{timestamp}.pkl', 'wb') as f:
        pickle.dump(config, f)
    
    print("\n" + "="*50)
    print("Training completed successfully!")
    print(f"Total training time: {datetime.now() - start_time}")
    print("="*50)

def create_custom_cnn():
    """Create a custom CNN model as fallback"""
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Rescaling(1./255),
        
        # First convolutional block
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second convolutional block
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third convolutional block
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth convolutional block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    
    return model

if __name__ == "__main__":
    main()