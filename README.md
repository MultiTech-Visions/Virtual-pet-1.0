# Festival Pet for Reachy Mini Wireless

An **offline, non-verbal virtual pet** for the Reachy Mini Wireless. No cloud, no LLM,
no speech. It beeps, boops, purrs, follows faces, remembers the people it has met,
and gets more affectionate with them over time. Pick it up and it snuggles; shake it
and it gets dizzy; leave it alone and it gets lonely, then dozes off.

Everything runs on the robot's own Raspberry Pi CM4. The only time it needs internet
is the one-time setup below.

## What it does

| Sense | Source | Behaviour |
|---|---|---|
| Faces | camera, YuNet at 320 px (~8 Hz) | locks onto the closest face, looks at it, micro-reacts every few seconds |
| Who is it | SFace embeddings, only on new tracks / every 4 s | stranger → curious "oh? hi!", friend → happy trill + wiggle, bestie → fanfare + a library move |
| Picked up | IMU accel/gyro in the base | startle, then purrs and snuggles; shaking → dizzy wobble |
| Petting | antenna pushed off its commanded angle | giggle + antenna wiggle, counts as affection for the person in front |
| Loud voice after quiet | mic array direction-of-arrival | perks up and looks toward it (heavily rate-limited: festivals are loud) |
| Time | – | energy drains while awake, refills asleep; lonely after 90 s alone, sleeps after 7 min |

Relationship memory is a JSON file of anonymous face embeddings plus stats (visits,
attention seconds, pets, holds, affection). Two visits apart by 2 minutes makes a
"friend"; four visits or lots of cuddling makes a "bestie". Nobody's name or photo is stored.

All sounds are synthesised in `festival_pet/sounds.py` (chirps, warbles, purrs) with
random variation so it never repeats itself exactly. Motion is layered in
`festival_pet/motion.py`: breathing + gaze + short cartoon gestures, plus optional
full-body moves from Pollen's emotions library for big moments.

## Layout

```
festival_pet/
  main.py       robot glue: 50 Hz control loop, sound thread, status page (ReachyMiniApp)
  behavior.py   state machine + mood (pure Python, unit-tested)
  motion.py     pose composition and gestures (numpy/scipy, unit-tested)
  vision.py     YuNet detection, single-target tracking, SFace recognition thread
  memory.py     persistent face memory / relationship stats
  senses.py     pickup, shake, antenna-touch, loud-sound detectors
  sounds.py     procedural droid vocalisations
  static/       tiny status page served at http://reachy-mini.local:8042
scripts/setup_offline.sh   one-time install + model/move-library download on the robot
tests/                     pytest suite for everything that does not need the robot
```

## One-time setup (robot online)

From your laptop, on the same Wi-Fi as the robot:

```bash
scp -r . pollen@reachy-mini.local:/home/pollen/festival_pet
ssh pollen@reachy-mini.local 'bash /home/pollen/festival_pet/scripts/setup_offline.sh'
```

The script installs the app into the daemon's apps venv (`/venvs/apps_venv`), downloads
the two OpenCV Zoo ONNX models (~39 MB) into `~/.local/share/festival_pet/models/`, caches
the `pollen-robotics/reachy-mini-emotions-library` dataset, and runs a load check.

Then open the dashboard at `http://reachy-mini.local:8000`, find **festival_pet** in the
installed apps and start it. The status page is at `http://reachy-mini.local:8042`.

## Auto-start at the festival

The daemon can launch one app on wake-up. On the robot:

```bash
sudo nano /venvs/mini_daemon/lib/python3.12/site-packages/reachy_mini/daemon/app/services/wireless/launcher.sh
```

Append `--startup-app festival_pet` to the `python -u -m reachy_mini.daemon.app.main` line, then
`sudo systemctl restart reachy-mini-daemon`. With the daemon's default `--no-wake-up-on-start`,
the robot boots asleep and **touching an antenna wakes it and launches the pet**.

(Alternative that survives daemon updates: the daemon also persists a startup app in
`~/.config/reachy_mini/daemon_config.json` under the key `startup_app`.)

## Festival operations

- Battery: ~30 min unplugged. Keep it on USB-C power at your spot; unplug for photos/cuddles.
  The app does not need to be restarted after replugging.
- The app sets `HF_HUB_OFFLINE=1` so nothing ever tries the network.
- "Forget everyone" button on the status page wipes the memory file.
- CPU: detection ~25–30 ms per frame on the CM4, one recognition ~250–350 ms and rare. If
  the head feels sluggish, raise `DETECT_INTERVAL` in `vision.py`.
- Set `PLAY_LIBRARY_SOUNDS = True` in `main.py` to hear Pollen's sidecar sounds with the
  library moves instead of the pet's own beeps.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The behavior engine takes a fake clock, so you can script whole interactions in tests
(see `tests/test_behavior.py`). The `festival_pet.vision.Vision.process_frame` method can be
driven with any BGR numpy frame off-robot.

## Credits and sources

- Reachy Mini SDK by Pollen Robotics (Apache-2.0): daemon, `look_at_image_pose`, recorded moves.
- Face models from the OpenCV Zoo (Apache-2.0): YuNet (`face_detection_yunet_2023mar.onnx`) and
  SFace (`face_recognition_sface_2021dec.onnx`).
- Behaviour ideas borrowed from community apps (desk_pet_bird, reachy_baby_yoda, recognizer,
  Reachy-companion): energy/social drives, priority-preempting gestures, SFace-every-N-frames,
  DoA startle.
