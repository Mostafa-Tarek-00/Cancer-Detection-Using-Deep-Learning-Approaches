import tensorflow as tf
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np


def train_model(model, X_train, Y_train, EPOCHS=25, save_path="model_weightsk.h5"):
    early_stop = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
    
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto')

    history = model.fit(X_train,
                        Y_train,
                        validation_split=0.2,
                        batch_size=64,
                        epochs=EPOCHS,
                        callbacks=[reduce_lr, early_stop])

    model.save_weights(save_path)

    return history

def test_model(model, X_train, Y_train, X_test, Y_test, label_mapping, data):
    test_acc = model.evaluate(X_test, Y_test, verbose=0)[1]
    print("Test Accuracy: {:.3f}%".format(test_acc * 100))
    
    train_acc = model.evaluate(X_train, Y_train, verbose=0)[1]
    print("Train Accuracy: {:.3f}%".format(train_acc * 100))

    y_true_test = np.array(Y_test)
    y_pred_probs_test = model.predict(X_test)
    y_pred_test = np.array(list(map(lambda x: np.argmax(x), y_pred_probs_test)))

    precision_test = precision_score(y_true_test, y_pred_test, average='weighted')
    recall_test = recall_score(y_true_test, y_pred_test, average='weighted')
    f1_test = f1_score(y_true_test, y_pred_test, average='weighted')
    
    print("\nTest Metrics:")
    print("Accuracy: {:.3f}".format(test_acc))
    print("Precision: {:.3f}".format(precision_test))
    print("Recall: {:.3f}".format(recall_test))
    print("F1 Score: {:.3f}".format(f1_test))

    y_true_train = np.array(Y_train)
    y_pred_probs_train = model.predict(X_train)
    y_pred_train = np.array(list(map(lambda x: np.argmax(x), y_pred_probs_train)))

    precision_train = precision_score(y_true_train, y_pred_train, average='weighted')
    recall_train = recall_score(y_true_train, y_pred_train, average='weighted')
    f1_train = f1_score(y_true_train, y_pred_train, average='weighted')
    
    print("\nTrain Metrics:")
    print("Accuracy: {:.3f}".format(train_acc))
    print("Precision: {:.3f}".format(precision_train))
    print("Recall: {:.3f}".format(recall_train))
    print("F1 Score: {:.3f}".format(f1_train))
    
    for i in range(len(X_test)):
        image_name = data['image_id'].iloc[i]
        true_label = label_mapping[y_true_test[i][0]]
        pred_label = label_mapping[y_pred_test[i]]
        confidence = y_pred_probs_test[i][y_pred_test[i]]

        print(f"Image: {image_name}, True Label: {true_label}, Predicted Label: {pred_label}, Confidence: {confidence:.4f}")
        
    clr_test = classification_report(y_true_test, y_pred_test, target_names=label_mapping.values())
    print("\nTest Classification Report:")
    print(clr_test)

    cm_test = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', xticklabels=label_mapping.values(), yticklabels=label_mapping.values())
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    sample_data_test = X_test[:15]
    plt.figure(figsize=(22, 12))

    for i in range(15):
        image_name = data['image_id'].iloc[i]
        plt.subplot(3, 5, i + 1)
        plt.imshow(sample_data_test[i])
        true_label_test = label_mapping[y_true_test[i][0]]
        pred_label_test = label_mapping[y_pred_test[i]]
        confidence_test = np.max(model.predict(np.expand_dims(sample_data_test[i], axis=0)))
        plt.title(f"{true_label_test} | {pred_label_test}\nConfidence: {confidence_test:.2f}")
        plt.axis("off")
        print(f"Confidence: {confidence_test:.2f}")
    plt.show()
    
def plot_model_training_curve(history):
    fig = make_subplots(rows=1, cols=2, subplot_titles=['Model Accuracy', 'Model Loss'])
    fig.add_trace(
        go.Scatter(
            y=history.history['accuracy'], 
            name='train_acc'), 
        row=1, col=1)
    fig.add_trace(
        go.Scatter(
            y=history.history['val_accuracy'], 
            name='val_acc'), 
        row=1, col=1)
    fig.add_trace(
        go.Scatter(
            y=history.history['loss'], 
            name='train_loss'), 
        row=1, col=2)
    fig.add_trace(
        go.Scatter(
            y=history.history['val_loss'], 
            name='val_loss'), 
        row=1, col=2)

    fig.update_xaxes(title_text='Epochs', row=1, col=1)
    fig.update_yaxes(title_text='Accuracy', row=1, col=1)
    fig.update_xaxes(title_text='Epochs', row=1, col=2)
    fig.update_yaxes(title_text='Loss', row=1, col=2)

    fig.show()