import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
from data.pamap2.pamap2 import create_pamap2

def plot_training_loss_balancing(H, plotPath):
    	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["classified_loss"], label="train_loss")
	plt.plot(H.history["val_classified_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	if plotPath is not None:
		plt.savefig(plotPath)
	else:
		plt.show()

def plot_training_acc_balancing(H, plotPath):
    	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["classified_accuracy"], label="train_accuracy")
	plt.plot(H.history["val_classified_accuracy"], label="val_accuracy")
	plt.title("Training accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("accuracy")
	plt.legend(loc="lower left")
	if plotPath is not None:
		plt.savefig(plotPath)
	else:
		plt.show()

def plot_training_loss(H, plotPath):
    	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["loss"], label="train_loss")
	plt.plot(H.history["val_loss"], label="val_loss")
	plt.title("Training Loss")
	plt.xlabel("Epoch #")
	plt.ylabel("Loss")
	plt.legend(loc="lower left")
	if plotPath is not None:
		plt.savefig(plotPath)
	else:
		plt.show()

def plot_training_acc(H, plotPath):
    	# construct a plot that plots and saves the training history
	plt.style.use("ggplot")
	plt.figure()
	plt.plot(H.history["accuracy"], label="train_accuracy")
	plt.plot(H.history["val_accuracy"], label="val_accuracy")
	plt.title("Training accuracy")
	plt.xlabel("Epoch #")
	plt.ylabel("accuracy")
	plt.legend(loc="lower left")
	if plotPath is not None:
		plt.savefig(plotPath)
	else:
		plt.show()

def print_test_result_balancing(model, filepath, x_test, y_classified, y_contrastive, dataset="wisdm"):
    model.load_weights(filepath)
    test_results = model.evaluate([x_test], [y_classified, y_contrastive])
    y_pred = model.predict([x_test])[0]
    matrix = confusion_matrix(y_classified, y_pred.argmax(axis=1))

    print("test acc  : ", test_results[3])
    print("test loss : ", test_results[0])

    score = f1_score(y_classified, y_pred.argmax(axis=1), average="macro")
    print("f1 score  : ", score)

    print(matrix)
    fontdict={'fontsize': 12}
    if dataset == "wisdm":
        label = ["Jogging", "LyingDown", "Sitting", "Stairs", "Stading", "Walking"]
    elif dataset == "pamap2":
        label = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs', 'vacuum_cleaning', 'ironing', 'rope_jumping']
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(label)))
    ax.set_xticklabels(label)
    ax.set_yticks(np.arange(len(label)))
    ax.set_yticklabels(label)
    plt.xticks(rotation = 90)
    
    ax.set_xlabel("Predicted label", **fontdict)
    ax.set_ylabel("True label", **fontdict)
    
    plt.show()

def print_test_result(model, filepath, x_test, y_classified, dataset="wisdm"):
    model.load_weights(filepath)
    test_results = model.evaluate(x_test, y_classified)
    y_pred = model.predict([x_test])[0]
    matrix = confusion_matrix(y_classified, y_pred.argmax(axis=1))

    print("test acc  : ", test_results[1])
    print("test loss : ", test_results[0])

    score = f1_score(y_classified, y_pred.argmax(axis=1), average="macro")
    print("f1 score  : ", score)

    print(matrix)
    fontdict={'fontsize': 12}
    if dataset == "wisdm":
        label = ["Jogging", "LyingDown", "Sitting", "Stairs", "Stading", "Walking"]
    elif dataset == "pamap2":
        label = ['lying', 'sitting', 'standing', 'walking', 'running', 'cycling', 'Nordic_walking', 'ascending_stairs', 'descending_stairs', 'vacuum_cleaning', 'ironing', 'rope_jumping']
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(label)))
    ax.set_xticklabels(label)
    ax.set_yticks(np.arange(len(label)))
    ax.set_yticklabels(label)
    plt.xticks(rotation = 90)
    
    ax.set_xlabel("Predicted label", **fontdict)
    ax.set_ylabel("True label", **fontdict)
    
    plt.show()