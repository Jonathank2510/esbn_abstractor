import tensorflow as tf
import tensorflow_datasets as tfds

def generate_data(train_batch_size=32, test_batch_size=32):
    train_raw, test_raw = tfds.load("scan/addprim_jump", split=["train", "test"])

    def standardize(text):
        text = tf.strings.join(["<SOS>", text, "<EOS>"], separator=' ')
        return text


    command_processor = tf.keras.layers.TextVectorization(
        standardize=standardize
    )
    action_processor = tf.keras.layers.TextVectorization(
        standardize=standardize
    )
    
    def select_fn(word):
        def select_word(input):
            return input[word]
        return select_word

    command_processor.adapt(train_raw.map(select_fn("commands")))
    action_processor.adapt(train_raw.map(select_fn("actions")))

    def process_scan(pair):
        command = command_processor(pair["commands"])
        action = action_processor(pair["actions"])
        action_in = action[:, :-1]
        action_out = action[:, 1:]
        return (command, action_in), action_out

    train_raw = train_raw.batch(train_batch_size, drop_remainder=True).prefetch(20)
    test_raw = test_raw.batch(test_batch_size, drop_remainder=True).prefetch(20)

    train_ds = train_raw.map(process_scan, tf.data.AUTOTUNE)
    val_ds = test_raw.map(process_scan, tf.data.AUTOTUNE)

    return train_ds, val_ds, len(command_processor.get_vocabulary()), len(action_processor.get_vocabulary())