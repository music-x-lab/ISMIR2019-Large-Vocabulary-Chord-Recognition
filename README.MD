# ISMIR Chord Recognition Code

Warning: uncleaned code

## Training

First prepare the jams dataset in the following format:

```
    chord_data_1217/
        audio/
            TR6R91L11C8A40D710.mp3
            ...
        chordlab/
            TR6R91L11C8A40D710.lab
            ...
```

Then modify the value of JAM_DATASET_PATH in ``settings.py`` to the path of the dataset (e.g., ``some_path/chord_data_1217``)

Then run ``storage_creation.py`` to get h5 data files ``jams_xchord.h5d`` and ``jams_cqt.h5d``.

Then run ``chordnet_ismir_naive.py 0`` for training/testing on data split #0.

If you encounter with some errors with ``torch.bool()`` 
just follows the error message to add some ``bool()`` to index tensors.

## Testing

Run ``chordnet_ismir_naive_eval.py`` for testing.

More docs TBA!

Junyan