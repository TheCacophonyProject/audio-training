# addtracks.py

Script that runs over a directory of recordings, in the same way as `perchembed.py` and
`birdnetanalyse.py`. It fills in whatever metadata a recording is missing and skips
anything that is already done. Each step below checks a version key and only runs when
that key is missing or differs from the current version, so the script can be re-run
safely. `--overwrite` only redoes the cacophony model predictions.

Metadata must follow the format of
`spectv10/1418778-20221221-072241-Alices Cacophonometer.txt`
(JSON in a `.txt` file next to the audio file, with the same stem).

## Input

- `dir`: searched recursively for audio (`.flac .wav .m4a .mp3 .ogg`).
- The `.txt` with the same stem is the metadata. If there is none, create it.
- Audio is loaded and resampled to 48 kHz (`load_recording`).

## Steps, per recording

1. **Metadata.** Load the `.txt` file. If there isn't one, create it with only
   `id` (the file stem) and `duration`. Set `file`.
2. **Signals.** If `signal_version` is missing or below 2.0, run
   `identifytracks.signal_noise` with its defaults (the same settings as audio-analysis)
   on the audio up to `get_end`. Save `signal` as
   `[start, end, freq_start, freq_end, strength]` (2 dp), plus `rec_end` and
   `signal_version: 2.0`. Version 2.0 marks the switch from `otherdata.signal_noise`,
   so all existing signal data gets redone.
3. **Tracks.** Only if the recording has no `Tracks`: build them with
   `identifytracks.get_tracks_from_signals(signals, rec_end)`. Give each track an id and
   save it in the same shape as the existing `Tracks` entries
   (`id, start, end, minFreq, maxFreq, positions`, marked `automatic: true`).
   Existing manual `Tracks` are never changed, and the models run on them instead.
4. **RMS.** If `rms_version` is missing or less than `RMS_VERSION`, run
   `otherdata.add_rms_data_to_tracks` on the tracks and set `rms_version`.
   This matches `otherdata.process_rms`.
5. **Model windows.** Each model is run on windows of its own context length, read from
   the model itself rather than hard-coded (BirdNET 3.0 is 3 s, Perch is 5 s, and the
   cacophony model's `segment_length` is 3 s):
   - If a track is shorter than the context, centre one window on the track and clamp
     it to stay inside the recording.
   - If a track is longer, use the fewest windows that cover it with at least 1 s of
     overlap, spread evenly from the track start to the track end.
6. **BirdNET.** For each track whose `birdnet_model` is missing or differs from the
   current model id, run BirdNET on the track's windows using the geo species list, as
   `birdnetanalyse.py` does. Save the detections inside the track as `track["birdnet"]`,
   with `track["birdnet_model"]` (e.g. `birdnet-3.0-pt`). Set `birdnet_geo` at the top
   level. The existing top-level whole-recording `birdnet` list is left alone.
7. **Perch embeddings.** The output folder holds the shard `.npy` and `.csv` files, in
   the same layout as `perchembed.py`. For each window, check the csv for
   `recording_id + track_id + perch_start`, and only embed windows that are missing.
   `recording_id` is the metadata `id`, which is the file stem for new metadata.
   Record the perch model version in the csv.
8. **Cacophony model.** If there are no results from the current model, load
   `~/cacophony/audio-model/spectv10/spectv10NoMixKokakoFine` with
   `val_pr_auc.weights.h5`. The model makes its own mel spectrogram, so its input is
   raw 48 kHz audio, peak-normalised to [-1, 1] as in training. The window length
   comes from the input size (144000 samples = 3 s), and the stride is
   `segment_stride` from `metadata.txt` (1 s). Run it on each track. For each track,
   save every label's score for each window (no threshold) in
   `track["cacophony-model"]`, with `model_name` and `weights`. These two are also
   what the skip check compares against. The labels are `ebird_labels` from
   `metadata.txt`, in the order of the model's outputs.

## Usage

    python addtracks.py <dir> --out <perch shard dir>

`--no-birdnet`, `--no-perch` and `--no-cacophony` skip a model. `--overwrite` reruns
the cacophony model on every track and replaces all of its earlier results. The other
steps are never redone once they're there.

## Versions saved

`signal_version`, `rms_version`, `birdnet_model`, the perch model in the csv, and
`model_name` for the cacophony model (plus the weights file).

