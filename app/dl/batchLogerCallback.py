from tensorflow.keras.callbacks import Callback

class BatchLossLogger(Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []

    def on_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))