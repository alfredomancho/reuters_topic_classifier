#!/usr/bin/env python3

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.datasets import reuters
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.models import load_model
from sklearn.isotonic import IsotonicRegression

# print Tensorflow version
print(tf.__version__)

# =====================
# Configuration
# =====================
VOCAB_SIZE = 20000       # Max vocabulary size
NUM_WORDS = VOCAB_SIZE
NUM_CLASSES = 46         # 46 topics in Reuters dataset
MAX_LEN = 300            # Max sequence length for padding
EMBED_DIM = 128
BATCH_SIZE = 64          # Samples per batch
EPOCHS = 8               # Number of training passes
VAL_SPLIT = 0.2          # Validation split
MODEL_FILE = 'reuters_topic_model.keras'
LOSS_PLOT_PATH = 'loss_curves.png'
CAL_PLOT_PATH = 'cal_curve.png'
CAL_PLOT_PATH_TEMPCAL = 'cal_curve_tempCal.png'
CAL_PLOT_PATH_ISO = 'cal_curve_tempCal_iso.png'

# Full list of 46 topic names (indices 0 through 45)
topic_labels = [
    'cocoa',         #  0
    'grain',         #  1
    'veg-oil',       #  2
    'earn',          #  3
    'acq',           #  4
    'wheat',         #  5
    'copper',        #  6
    'housing',       #  7
    'money-supply',  #  8
    'coffee',        #  9
    'sugar',         # 10
    'trade',         # 11
    'reserves',      # 12
    'ship',          # 13
    'cotton',        # 14
    'carcass',       # 15
    'crude',         # 16
    'nat-gas',       # 17
    'cpi',           # 18
    'money-fx',      # 19
    'interest',      # 20
    'gnp',           # 21
    'meal-feed',     # 22
    'alum',          # 23
    'oilseed',       # 24
    'gold',          # 25
    'tin',           # 26
    'strategic-metal',# 27
    'livestock',     # 28
    'retail',        # 29
    'ipi',           # 30
    'iron-steel',    # 31
    'rubber',        # 32
    'heat',          # 33
    'jobs',          # 34
    'lei',           # 35
    'bop',           # 36
    'zinc',          # 37
    'orange',        # 38
    'pet-chem',      # 39
    'dlr',           # 40
    'gas',           # 41
    'silver',        # 42
    'wpi',           # 43
    'hog',           # 44
    'lead'           # 45
]

def decode_topic(label):
    return topic_labels[label]

def decode_newswire(seq):
    return " ".join(index_to_word.get(i, "?") for i in seq)

def build_model():

    model = Sequential([
    
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, input_length=MAX_LEN),
        Conv1D(filters=128, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax'),
    ])
    
    # Apply label smoothing after confirming "noisy" data 
    loss_fn = tf.keras.losses.CategoricalCrossentropy(
    label_smoothing=0.05, from_logits=False
    )

    # Compile model and train
    model.compile(
        loss=loss_fn,
        optimizer="adam",
        metrics=["accuracy"]
    )
    return model

def train_and_store(model, trainx, trainy):
    history = model.fit(
        trainx, trainy,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VAL_SPLIT
    )
    model.save(MODEL_FILE)
    return history

def extract_logits(model, data):

    # "Warm up" the model so that model.inputs exists
    _ = model.predict(data[:1])

    # Grab the final Dense layer which has `units=num_classes`, activation='softmax'
    final_dense = model.layers[-1]

    # Build a new model that has exactly the same weights,
    # but no softmax on the last layer (so we get logits).
    # Re-create the final Dense with activation=None:
    logits_layer = tf.keras.layers.Dense(
        final_dense.units,
        activation=None,
        name="logits_layer",
    )

    inp = model.inputs[0]
    x = inp
    for layer in model.layers[:-1]:
        x = layer(x)
    logits = logits_layer(x)  

    # Copy weights (kernel & bias) from the original softmax layer
    logits_layer.set_weights(final_dense.get_weights())

    # Create & run the small model
    logits_model = tf.keras.Model(inputs=inp, outputs=logits)
    return logits_model.predict(data)

def debug_test_set_predictions(model):
    # Get predicted probabilities for the test set
    probs = model.predict(x_test)            # shape (N, num_classes)
    preds = np.argmax(probs, axis=1)         # shape (N,)
    
    # Compute per‐sample cross‐entropy loss (no reduction)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.NONE
    )
    per_sample_losses = loss_fn(y_test, probs).numpy()  # shape (N,)

    # 3. Look at the worst‐offenders
    worst_idxs = np.argsort(per_sample_losses)[-5:][::-1]
    for i in worst_idxs:
        print(f"Index={i}  Loss={per_sample_losses[i]:.2f}")
        print(" Text:", decode_newswire(x_test[i])[:MAX_LEN], "…")
        print(f" True={y_test[i]}  Pred={preds[i]}  Prob(confidence)={probs[i,preds[i]]:.2f}")
        print(f"True label ({y_test[i]}):", decode_topic(y_test[i]))
        print(f"Pred label ({preds[i]}):", decode_topic(preds[i]))
        print("-" * 60)

    # Build confusion matrix (for debug)
    cm = tf.math.confusion_matrix(y_test, preds, num_classes=num_classes).numpy()
    print(cm)
    # visualize the row for class 18:
    print(f"Class 18 ('cpi') predicted as 3 ('earn'):", cm[18, 3])

    # Inspect full probability distribution
    probs = model.predict(x_test[i:i+1])[0]
    top_idxs = probs.argsort()[-5:][::-1]
    for idx in top_idxs:
        print(decode_topic(idx), f"{probs[idx]:.3f}")

def evaluate_model(model, test_x, test_y):
    # Evaluate on test set
    loss, acc = model.evaluate(test_x, test_y)
    print(f'Test Loss: {loss:.4f}  Test Accuracy: {acc:.4f}')

def plot_loss(history):
    """
    Plot training and validation loss curves and save to file.
    """
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(LOSS_PLOT_PATH)
    print(f'Loss curves saved to {LOSS_PLOT_PATH}')

def plot_reliability(bin_conf, bin_acc, plot_path):
    """
    Plot Expected Calibration Error (ECE) with a shaded ±5% band and save to file
    """
    # Plot reliability diagram with shaded ±5% band
    plt.figure(figsize=(6,6))
    plt.plot(bin_conf, bin_acc, marker='o', label='Reliability')
    plt.plot([0,1], [0,1], '--', label='Ideal')
    # shaded region
    plt.fill_between([0, 1], [0 - 0.05, 1 - 0.05], [0 + 0.05, 1 + 0.05], color='gray', alpha=0.2, label='±5% band')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.xlabel('Mean confidence')
    plt.ylabel('Empirical accuracy')
    plt.title('Reliability diagram')
    plt.legend()
    plt.savefig(plot_path)
    plt.show()
    print(f'Calibration plot saved to {plot_path}')

def calc_calibration_metrics(model, valx, valy):
    # Get model predictions
    probs = model.predict(valx)           # shape (N, K)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)
    accuracies  = (preds == valy).astype(float)

    # Bin confidences and compute per-bin accuracy and mean confidence
    bins = np.linspace(0, 1, 11)
    bin_ids = np.digitize(confidences, bins) - 1  # 0-indexed bins

    bin_acc = []
    bin_conf = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = (bin_ids == i)
        count = np.sum(mask)
        bin_counts.append(count)
        if count > 0:
            bin_acc.append(np.mean(accuracies[mask]))
            bin_conf.append(np.mean(confidences[mask]))
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)

    # Compute maximum deviation from ideal line
    deviations = np.abs(np.array(bin_acc) - np.array(bin_conf))
    max_dev = deviations.max()
    print(f"Maximum |accuracy - confidence| across bins: {max_dev:.4f}")

    # Return core metrics
    return confidences, accuracies, bin_conf, bin_acc, bin_counts, max_dev, preds

def calc_calibration_metrics_cal(model, cal_probs, valy):
    """
    Plot Expected Calibration Error (ECE) with a shaded ±5% band, save to file, and return metrics.
    """
    # Get model predictions
    preds = np.argmax(cal_probs, axis=1)
    confidences = np.max(cal_probs, axis=1)
    accuracies  = (preds == valy).astype(float)

    # Quantile binning for noisy Reuters dataset
    bins = np.percentile(confidences, np.linspace(0, 100, 11))
    bin_ids = np.digitize(confidences, bins) - 1

    bin_acc = []
    bin_conf = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = (bin_ids == i)
        count = np.sum(mask)
        bin_counts.append(count)
        if count > 0:
            bin_acc.append(np.mean(accuracies[mask]))
            bin_conf.append(np.mean(confidences[mask]))
        else:
            bin_acc.append(0.0)
            bin_conf.append(0.0)

    # Compute maximum deviation from ideal line
    deviations = np.abs(np.array(bin_acc) - np.array(bin_conf))
    max_dev = deviations.max()
    print(f"Maximum |accuracy - confidence| across bins: {max_dev:.4f}")

    # Return core metrics
    return confidences, accuracies, bin_conf, bin_acc, bin_counts, max_dev, preds

def iso_quant_bin(scaled_probs_val, scaled_probs_test, n_bins):

    # Gather validation‐set data for fitting
    probs_val   = np.max(scaled_probs_val, axis=1)   # temp‐scaled val confidences
    preds_val   = np.argmax(scaled_probs_val, axis=1)
    acc_val     = (preds_val == y_val).astype(float)

    # Fit isotonic on VAL
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(probs_val, acc_val)

    # Apply to TEST
    probs_test  = np.max(scaled_probs_test, axis=1)  # temp‐scaled test confidences
    preds_test  = np.argmax(scaled_probs_test, axis=1)
    acc_test    = (preds_test == y_test).astype(float)

    conf_iso_test = iso.transform(probs_test)        # calibrated test confidences

    # Compute ECE on TEST
    ece_iso = expected_calibration_error(conf_iso_test, acc_test, n_bins=10)

    # Compute max‐deviation on TEST with quantile bins
    edges = np.percentile(conf_iso_test, np.linspace(0,100,11))
    bin_ids = np.digitize(conf_iso_test, edges) - 1

    bin_conf = []
    bin_acc  = []
    for i in range(10):
        mask = (bin_ids == i)
        if mask.sum():
            bin_conf.append(conf_iso_test[mask].mean())
            bin_acc.append(acc_test[mask].mean())

    max_dev_iso = np.max(np.abs(np.array(bin_acc) - np.array(bin_conf)))

    return ece_iso, max_dev_iso, bin_conf, bin_acc

def fit_temp(logits, y_val):
    T = tf.Variable(1.0, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(1e-2)

    for _ in range(100):
        with tf.GradientTape() as tape:
            scaled_logits = logits / T
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_val, scaled_logits, from_logits=True
                )
            )
        grads = tape.gradient(loss, [T])
        optimizer.apply_gradients(zip(grads, [T]))

    print("Learned T:", T.numpy())
    return T

def expected_calibration_error(confidences, accuracies, n_bins=10):
    # confidences and accuracies are 1D arrays of length N
    bins      = np.linspace(0.0, 1.0, n_bins+1)
    bin_ids   = np.digitize(confidences, bins) - 1  # 0-indexed
    ece       = 0.0
    N         = len(confidences)

    for m in range(n_bins):
        # indices for bin m
        bin_mask = (bin_ids == m)
        bin_count = np.sum(bin_mask)
        if bin_count == 0:
            continue
        avg_conf = np.mean(confidences[bin_mask])
        avg_acc  = np.mean(accuracies[bin_mask])
        ece += (bin_count / N) * abs(avg_acc - avg_conf)

    return ece

def predict_new_sample(model, test_x, test_y):
    # pick a random test sample
    i = np.random.randint(len(test_x))
    sample_seq = test_x[i]
    print("Text:", decode_newswire(sample_seq))
    print("Actual label:", test_y[i])
    
    probs = model.predict(sample_seq.reshape(1, -1))[0]
    pred_label = np.argmax(probs)
    print("Predicted label:", pred_label, f"(confidence {probs[pred_label]:.2f})")







if __name__ == "__main__":
    # Load raw Reuters data
    (x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=NUM_WORDS)
    word_index = reuters.get_word_index()

    val_size = int(len(x_train) * VAL_SPLIT)

    x_val = x_train[:val_size]      # i.e. 0 to val_size (0.2)
    y_val = y_train[:val_size]

    x_train = x_train[val_size:]    # i.e. val_size to end
    y_train = y_train[val_size:]

    # One-hot encode labels (for crossentropy/label-smoothing)
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test_one_hot  = tf.keras.utils.to_categorical(y_test,  NUM_CLASSES)
    y_val_one_hot  = tf.keras.utils.to_categorical(y_val,  NUM_CLASSES)

    # Reserve indices for padding/unknown tokens
    index_to_word = {index + 3: word for word, index in word_index.items()}
    index_to_word[0] = "<PAD>"
    index_to_word[1] = "<START>"
    index_to_word[2] = "<UNK>"
    index_to_word[3] = "<UNUSED>"

    # Peek at the first article
#    print(decode_newswire(x_train[0])[:300], "...")
#    print("Label:", y_train[0])

    # Pad sequence
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, padding="post", truncating="post")
    x_test  = pad_sequences(x_test,  maxlen=MAX_LEN, padding="post", truncating="post")
    x_val = pad_sequences(x_val, maxlen=MAX_LEN, padding="post", truncating="post")
    num_classes = max(y_train) + 1  

    # Build model
    topic_model = build_model()
    topic_model.summary()

    # Train and retrieve history
    history = train_and_store(topic_model, x_train, y_train_one_hot)  #one-hot for label-smoothing

    # Load trained model (for debug)
#    topic_model = load_model(MODEL_FILE) # .keras or .h5

    # debug_test_set_predictions
#    debug_test_set_predictions(topic_model)

    # Evaluate on test dataset
    evaluate_model(topic_model, x_test, y_test_one_hot)  #one-hot labels for label-smoothing

    # Plot and save loss curves
    plot_loss(history)

    # Extract logits to apply temperature scaling
    logits_val = extract_logits(topic_model, x_val)
    logits_test = extract_logits(topic_model, x_test)

    # Apply temperature scaling
    temp_val = fit_temp(logits_val, y_val)
    temp_test = fit_temp(logits_test, y_test)

    calibrated_probs_val = tf.nn.softmax(logits_val / temp_val, axis=1).numpy()
    calibrated_probs_test = tf.nn.softmax(logits_test / temp_test, axis=1).numpy()

    # Plot calibration/reliability curves on test set
    (conf, acc, bin_conf, bin_acc, bin_counts, max_dev, preds) = calc_calibration_metrics(topic_model, x_test, y_test)

    plot_reliability(bin_conf, bin_acc, CAL_PLOT_PATH)

    # Plot calibration/reliability curves applying temperature scaling on validation set
    (confCAL, accCAL, bin_confCAL, bin_accCAL, bin_countsCAL, max_devCAL, predsCAL) = calc_calibration_metrics_cal(topic_model, calibrated_probs_val, y_val)

    plot_reliability(bin_confCAL, bin_accCAL, CAL_PLOT_PATH_TEMPCAL)

    # Apply isometric regression on temp-scaled quantile bins on test set and plot results
    (ece_iso, max_dev_iso, bin_conf_iso, bin_acc_iso) = iso_quant_bin(calibrated_probs_val, calibrated_probs_test, 10) 

    plot_reliability(bin_conf_iso, bin_acc_iso, CAL_PLOT_PATH_ISO)

    # Display expected calibration error (ECE) and max deviation for baseline, temp-scaled quantile bins and after apply isotonic regression on test set
    print()
    print()
    ece_value = expected_calibration_error(conf, acc, n_bins=10)
    print(f"Expected Calibration Error (ECE): {ece_value:.4f}")
    print(f"Max Deviation: {max_dev:.4f}")
    print()

    ece_value_cal = expected_calibration_error(confCAL, (predsCAL == y_val).astype(float), n_bins=10)
    print(f"ECE after temperature scaling: {ece_value_cal:.4f}")
    print(f"Max Deviation after temperature scaling and quantile binning: {max_devCAL:.4f}")
    print()

    print(f"ECE on test set after isotonic: {ece_iso:.4f}")
    print(f"Max deviation on test set after isotonic: {max_dev_iso:.4f}")
    print()

    # Predict on New Samples
#    predict_new_sample(topic_model, x_test, y_test)


