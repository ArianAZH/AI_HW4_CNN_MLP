from sklearn.preprocessing import StandardScaler
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout,Flatten ,Conv2D, MaxPooling2D, BatchNormalization, Activation, Input, GlobalAveragePooling2D , SpatialDropout2D 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay , classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from keras.applications import ResNet50
from keras.optimizers import Adam
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET
import os


folder_path1 = "UCI HAR Dataset/UCI HAR Dataset"


Xtrain1 = np.loadtxt(f"{folder_path1}/train/X_train.txt")
ytrain1 = np.loadtxt(f"{folder_path1}/train/y_train.txt")
X_val1 = np.loadtxt(f"{folder_path1}/test/X_test.txt")
y_val1 = np.loadtxt(f"{folder_path1}/test/y_test.txt")

ytrain1 = ytrain1.astype(int) - 1
y_val1 = y_val1.astype(int) - 1



scaler = StandardScaler()
Xtrain1 = scaler.fit_transform(Xtrain1)
X_val1 = scaler.transform(X_val1)

X_train1, X_test1, y_train1, y_test1 = train_test_split(Xtrain1, ytrain1, test_size=0.15, random_state=42)


print("Train shape:", X_train1.shape)
print("Test shape:", X_test1.shape)


model1_mlp = Sequential([
    Dense(128, activation='relu', input_shape=(561,)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')  
])


X_train_cnn = X_train1.reshape(-1, 33, 17, 1)
X_test_cnn = X_test1.reshape(-1, 33, 17, 1)
y_train_cat = to_categorical(y_train1, num_classes=6)
y_test_cat = to_categorical(y_test1, num_classes=6)

X_val_cnn = X_val1.reshape(-1, 33, 17, 1)
y_val_cat = to_categorical(y_val1, num_classes=6)

model1_cnn = Sequential([

    Conv2D(32, (3, 3), padding='same', input_shape=(33, 17, 1)),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),


    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),


    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),


    Flatten(),
    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),


    Dense(6, activation='softmax')
])


model1_mlp.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_mlp = model1_mlp.fit(X_train1, y_train1, epochs=30, validation_data=(X_test1, y_test1))


model1_cnn.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history_cnn = model1_cnn.fit(X_train_cnn, y_train_cat, epochs=30, validation_data=(X_test_cnn, y_test_cat))



def plot_history(history, title):
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

plot_history(history_mlp, "MLP Performance")
plot_history(history_cnn, "CNN Performance")



loss_mlp1, acc_mlp1 = model1_mlp.evaluate(X_val1, y_val1, verbose=0)
print(f"accuracy on mlp test data  {acc_mlp1*100:.2f}%")


loss_cnn1, acc_cnn1 = model1_cnn.evaluate(X_test_cnn, y_test_cat, verbose=0)
print(f" accuracy on cnn test data {acc_cnn1*100:.2f}%")


y_pred_mlp = model1_mlp.predict(X_val1).argmax(axis=1)
y_pred_cnn = model1_cnn.predict(X_val_cnn).argmax(axis=1)

cm_mlp = confusion_matrix(y_val1, y_pred_mlp)
cm_cnn = confusion_matrix(y_val1, y_pred_cnn)

ConfusionMatrixDisplay(cm_mlp).plot()
plt.title("MLP Confusion Matrix")
plt.show()

ConfusionMatrixDisplay(cm_cnn).plot()
plt.title("CNN Confusion Matrix")
plt.show()

# ##2

def load_images_from_folders(base_dir, image_size=(200, 200)):
    images = []
    labels = []
    class_names = sorted(os.listdir(base_dir))  
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(base_dir, class_name)
        for fname in os.listdir(class_dir):
            img_path = os.path.join(class_dir, fname)
            try:
                img = Image.open(img_path).convert("L").resize(image_size)
                img = np.array(img) / 255.0  
                images.append(img)
                labels.append(class_to_idx[class_name])
            except Exception as e:
                print(f"error, image not found{img_path}: {e}")

    images = np.array(images).reshape(-1, image_size[0], image_size[1], 1)
    labels = np.array(labels)
    return images, labels, class_names


train_dir = 'NEU-DET Dataset/NEU-DET/train/images'
val_dir = 'NEU-DET Dataset/NEU-DET/validation/images'

Xtrain2, ytrain2, class_names = load_images_from_folders(train_dir)
X_val2, y_val2, _ = load_images_from_folders(val_dir)

X_train2, X_test2, y_train2, y_test2 = train_test_split(Xtrain2, ytrain2, test_size=0.15, random_state=42)


print("Train set:", X_train2.shape, y_train2.shape)
print("Validation set:", X_test2.shape, y_test2.shape)
print("Classes:", class_names)




model2_mlp = Sequential([
    Flatten(input_shape=(200, 200, 1)),

    Dense(256),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.4),

    Dense(128),
    BatchNormalization(),
    Activation('relu'),

    Dense(6, activation='softmax')
])

model2_mlp.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


model2_cnn = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])

model2_cnn.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history2_mlp = model2_mlp.fit(X_train2, y_train2, epochs=30, validation_data=(X_test2, y_test2))

history2_cnn = model2_cnn.fit(X_train2, y_train2, epochs=30, validation_data=(X_test2, y_test2))

model2_cnn2 = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(200, 200, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    SpatialDropout2D(0.4),  

    Conv2D(64, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    SpatialDropout2D(0.4),

    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    SpatialDropout2D(0.4),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),  
    Dense(6, activation='softmax')
])

model2_cnn2.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history2_cnn2 = model2_cnn2.fit(X_train2, y_train2, epochs=30, validation_data=(X_test2, y_test2))

model2_cnn_fact = Sequential([
    Conv2D(32, (3,1), padding='same', activation='relu', input_shape=(200, 200, 1)),
    Conv2D(32, (1,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,1), padding='same', activation='relu'),
    Conv2D(64, (1,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,1), padding='same', activation='relu'),
    Conv2D(128, (1,3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(6, activation='softmax')
])
model2_cnn_fact.compile(optimizer=Adam(0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history2_cnn_fact = model2_cnn_fact.fit(X_train2, y_train2, epochs=30, validation_data=(X_test2, y_test2))


plot_history(history2_mlp, "MLP Accuracy/Loss")
plot_history(history2_cnn, "CNN Accuracy/Loss")
plot_history(history2_cnn2, "CNN (Block Dropout)  Accuracy/Loss")
plot_history(history2_cnn_fact, "CNN (Kernel Factorization) Accuracy/Loss")



loss_mlp, acc_mlp = model2_mlp.evaluate(X_val2, y_val2, verbose=0)
print(f"accuracy on mlp test data  {acc_mlp*100:.2f}%")

loss_cnn, acc_cnn = model2_cnn.evaluate(X_val2, y_val2, verbose=0)
print(f" accuracy on cnn test data {acc_cnn*100:.2f}%")

loss_cnn, acc_cnn = model2_cnn2.evaluate(X_val2, y_val2, verbose=0)
print(f" accuracy on cnn (Block Dropout) test data {acc_cnn*100:.2f}%")

loss_cnn, acc_cnn = model2_cnn_fact.evaluate(X_val2, y_val2, verbose=0)
print(f" accuracy on cnn (Kernel Factorization) test data {acc_cnn*100:.2f}%")


def show_conf_matrix(model, X_test, y_test, title):
    y_pred = model.predict(X_test).argmax(axis=1)
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(xticks_rotation=45)
    plt.title(title)
    plt.show()

show_conf_matrix(model2_mlp, X_val2, y_val2, "MLP Confusion Matrix")
show_conf_matrix(model2_cnn, X_val2, y_val2, "CNN Confusion Matrix")
show_conf_matrix(model2_cnn2, X_val2, y_val2, "CNN (Block Dropout) Confusion Matrix")
show_conf_matrix(model2_cnn_fact, X_val2, y_val2, "CNN (Kernel Factorization) Confusion Matrix")


##3
train_dir = 'NEU-DET Dataset/NEU-DET/train/images'
val_dir = 'NEU-DET Dataset/NEU-DET/validation/images'


img_size = (224, 224)  
batch_size = 32


def load_dataset(base_dir):
    images = []
    labels = []
    class_names = sorted(os.listdir(base_dir))
    class_to_idx = {cls: i for i, cls in enumerate(class_names)}

    for cls in class_names:
        cls_dir = os.path.join(base_dir, cls)
        for fname in os.listdir(cls_dir):
            img_path = os.path.join(cls_dir, fname)
            img = Image.open(img_path).convert('RGB').resize(img_size)
            images.append(np.array(img))
            labels.append(class_to_idx[cls])

    return np.array(images), np.array(labels), class_names

X_train, y_train, class_names = load_dataset(train_dir)
X_val, y_val, _ = load_dataset(val_dir)

print("Train:", X_train.shape, "Val:", X_val.shape)


X_train = X_train / 255.0
X_val = X_val / 255.0


train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator()  

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)


for i, class_name in enumerate(class_names):
    idx = np.where(y_train == i)[0][0]
    plt.imshow(X_train[idx])
    plt.title(f"Class: {class_name}")
    plt.axis("off")
    plt.show()


base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))


for layer in base_model.layers:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
output = Dense(6, activation='softmax')(x)

model_resnet = Model(inputs=base_model.input, outputs=output)
model_resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model_resnet.summary()


history_resnet_head = model_resnet.fit(train_generator, validation_data=val_generator, epochs=10)



for layer in base_model.layers[-20:]: 
    layer.trainable = True


model_resnet.compile(optimizer=Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


history_resnet_ft = model_resnet.fit(train_generator, validation_data=val_generator, epochs=10)



y_pred = model_resnet.predict(X_val).argmax(axis=1)


loss, acc = model_resnet.evaluate(X_val, y_val)
print(f"Accuracy: {acc:.4f} - Loss: {loss:.4f}")


cm = confusion_matrix(y_val, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix - ResNet50")
plt.show()


print(classification_report(y_val, y_pred, target_names=class_names))
