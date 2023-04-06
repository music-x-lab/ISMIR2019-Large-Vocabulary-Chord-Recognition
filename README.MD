# Large-Vocabulary Chord Transcription via Chord Structure Decomposition

This is the official repo for the ISMIR 2019 paper [Large-Vocabulary Chord Transcription via Chord Structure Decomposition](https://archives.ismir.net/ismir2019/paper/000078.pdf).

## Chord recognition with pretrained model

After installing all the dependencies, run the following code:

```
python3 chord_recognition.py path_to_audio_file path_to_output_file [chord_dict]
```

For example,

```
python3 chord_recognition.py example.mp3 example_chord.lab
```

Here, ``chord_dict`` is an optional parameter that tells the HMM which chord dictionary to use for decoding. Potential options are:

* submission: the default value (recommended to use). It is the chord dictionary we used for ISMIR 2019 submission.
* ismir2017: the chord dictionary for MIREX competition.
* full: the list of all chords from MARL dataset. It is not tested and not recommended to use. 

You may also manually adjust the chord dictionary by editing them in the folder ``data/*_chord_list.txt``.

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

Junyan
