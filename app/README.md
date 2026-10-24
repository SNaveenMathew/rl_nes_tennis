# Tennis RL Label Studio

A local Streamlit app for reviewing NES Tennis demonstrations and managing task/state annotations. It does not alter `.npz` demonstrations. The app keeps its working labels, review status, per-row notes/confidence, and audit history in a local SQLite database which is ignored by Git.

## Start it on Windows

From the repository root, in the active `tennis_rl` environment:

```powershell
python -m pip install -r app/requirements.txt
python -m streamlit run app/annotation_app.py
```

The app opens in your browser. It defaults to `demonstrations/`; choose another folder in the left sidebar if needed. Files are discovered using the usual `level_*/<outcome>/match_*/point_*.npz` hierarchy.

The video preview encoder uses `imageio-ffmpeg` (included in `app/requirements.txt`). Install dependencies into the same environment that launches Streamlit; using `python -m streamlit` helps keep the app and `pip` tied to the same interpreter. If `imageio-ffmpeg` is unavailable but system `ffmpeg` is installed and on `PATH`, the app uses that as a fallback.

## Reviewing and editing task segments

1. Select the point in **Point queue**. Existing `task_annotations.jsonl` labels and `state_annotations.jsonl` state rows are imported into `app/data/annotations.sqlite3` on the first run for each demonstration root.
2. Use the **Video playback** controls to play/pause and seek through an MP4 generated from the recorded frames. The **Exact step preview** slider displays one specific saved observation and its recorded action/pulse metadata.
3. Edit any task row directly in the table. `Start` is inclusive and `End` is exclusive, so `20,28` labels steps 20–27. Valid, complete task-table edits are automatically saved to SQLite; the summary count refreshes after saving. Use the `+` row control to add rows and select the trash icon on a row to delete it. You can also use the start/end inputs and **Add segment** to add and save an interval from scratch or use the preview step as its start.
4. **Save edits** remains available as an explicit commit/retry button. The app validates label names, boundaries, allowed manager durations (`1, 2, 4, 8`), and overlapping task intervals. Invalid or incomplete ranges are not saved; they remain as a per-point draft and are restored if you navigate away and return.
5. Set point review status and notes and click **Save point review status**. Available states: Not started, In progress, Needs review, Reviewed, Rejected.
6. When ready for existing training scripts, click **Export task labels to task_annotations.jsonl** (or the per-point export button). This writes the SQLite task labels into the JSONL format expected by the current training pipeline. The app database itself is not consumed by training.

## State/event annotations

Expand **Optional per-frame state / event labels** to display and edit the rows loaded from `state_annotations.jsonl`. Use one row per saved step. Coordinate fields are JSON pairs; `ball_xy` and `player_xy` must be normalized to `[0, 1]`. `events_json` is a JSON object, e.g. `{"agent_contact_window": 1}`; `1=yes`, `0=no`, omitted means unknown. Save the edits and export the state labels when ready.

## Import/export behavior

- Existing JSONL files are imported once per dataset root into SQLite. This avoids overwriting DB edits on ordinary app reruns.
- **Merge labels from JSONL** adds previously unseen rows without replacing existing DB labels. It does not overwrite a point's edited/deleted SQLite rows.
- **Export** atomically replaces the JSONL file with the app database's contents for that demo root. Do this before training if you changed labels in the app.
- The DB is stored at `app/data/annotations.sqlite3`, generated MP4 previews are under `app/.cache/videos/`, and both are Git-ignored.
- The database is scoped by the resolved demo-root path, so different data folders do not share annotation rows accidentally.

## Practical notes

- `--sample-every` from the CLI annotation script is not required here: the app uses the point's saved frames and step indices directly.
- Video playback uses `frame_skip` from the point's JSON sidecar. If it is missing, the app warns and assumes one raw NES frame per saved step for playback timing. This assumption does not change saved observations or annotations.
- Older points without `native_frames` are displayed using their stored grayscale observation. Lost detail cannot be recovered by scaling.
- This is designed for a single local annotator / small trusted team using a shared machine. SQLite uses transactions and a 15-second busy timeout, but it is not a hosted multi-user annotation service with authentication or access control.
