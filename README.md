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
| Its name | Vosk keyword spotting, only while the mic array flags speech | "Reachy!" → "huh? me?" chirp, perks up, turns toward the voice, listens for a trick for 8 s |
| Tricks | same grammar: `dance`, `hello`/`hi`, `good`, `sleep` | "Reachy, dance" → 6 s little groove; a second "dance" within 15 s → a lively library dance |
| Belly scratch | fingernail clicks on the shell, heard by the mics | ticklish giggle + wiggle, counts as a pet |
| Music | beat tracker on the mic stream (spectral flux + autocorrelation) | subtle head bob and antenna sway on the beat, three groove styles, the odd "sing-along" blip |
| Peekaboo | face hidden 0.6–3.5 s then back | giggle + bounce |
| Being stared at | same face very close for 14 s | goes shy: looks away, antennas fold, peeks back |
| Empathy | the person's head tilt (eye line) | slowly mirrors the tilt |
| Time | – | energy drains while awake, refills asleep; lonely after 90 s alone, nods off, sleeps after 7 min; rare sneezes and hiccups |

Relationship memory is a JSON file of anonymous face embeddings plus stats (visits,
attention seconds, pets, holds, affection). Two visits apart by 2 minutes makes a
"friend"; four visits or lots of cuddling makes a "bestie". Nobody's name or photo is stored.

All sounds are synthesised in `festival_pet/sounds.py` (chirps, warbles, purrs) with
random variation so it never repeats itself exactly. Motion is layered in
`festival_pet/motion.py`: breathing + gaze + short cartoon gestures, plus optional
full-body moves from Pollen's emotions library for big moments.

### Notes on the harder senses

- **Name**: "reachy" is not in Vosk's English lexicon, so the grammar spots in-vocabulary
  sound-alikes ("ricci", "richie", "reach it"…) that fire on the spoken name, with decoys
  ("peachy", "reach", "beach") to absorb near-misses. Verified on synthesized speech in
  several voices (`tests/test_hearing.py`). Recognition only runs while the ReSpeaker flags
  speech, so it costs nothing while music plays.
- **Belly scratch**: fingernails on the shell reach the mics as structure-borne clicks that
  are very short, broadband and strong above 3 kHz. The detector wants 4+ such clicks within
  1.2 s that each decay within ~30 ms (consonants and hi-hats ring longer). Tested against
  synthetic clicks over music and against speech clips; **the thresholds in
  `audio_features.ScratchDetector` will need a tuning pass on the real shell** (watch
  `band_energy` vs `median` on the status page while you scratch).
- **Beat tracking**: 60–180 BPM, needs ~4 s of stable tempo before the pet starts grooving,
  and it stops when the beat goes away. Intensity scales with energy; it never does big moves
  on its own, the lively dance only happens when asked twice.

## Layout

```
festival_pet/
  main.py       robot glue: 50 Hz control loop, sound thread, status page (ReachyMiniApp)
  behavior.py   state machine + mood (pure Python, unit-tested)
  motion.py     pose composition and gestures (numpy/scipy, unit-tested)
  vision.py     YuNet detection, single-target tracking, SFace recognition thread
  memory.py     persistent face memory / relationship stats
  senses.py     pickup, shake, antenna-touch, loud-sound detectors
  audio_features.py  beat tracker + shell-scratch detector (mic stream)
  hearing.py    Vosk name / trick-word spotting
  sounds.py     procedural droid vocalisations
  static/       tiny status page served at http://reachy-mini.local:8042
scripts/setup_offline.sh   one-time install + model/move-library download on the robot
scripts/sim_harness.py     drives the whole pet against the SDK's MuJoCo simulator
tests/                     pytest suite for everything that does not need the robot
```

## One-time setup (robot online)

From your laptop, on the same Wi-Fi as the robot:

```bash
scp -r . pollen@reachy-mini.local:/home/pollen/festival_pet
ssh pollen@reachy-mini.local 'bash /home/pollen/festival_pet/scripts/setup_offline.sh'
```

The script installs the app into the daemon's apps venv (`/venvs/apps_venv`), downloads
the two OpenCV Zoo ONNX models (~39 MB) and the Vosk small English model (~40 MB) into
`~/.local/share/festival_pet/models/`, caches the
`pollen-robotics/reachy-mini-emotions-library` dataset, and runs a load check.

Then open the dashboard at `http://reachy-mini.local:8000`, find **festival_pet** in the
installed apps and start it. The pet's own page ("Reachy's Mind") is at
`http://reachy-mini.local:8042`.

## Wi‑Fi at the festival: let the robot be the hotspot

The wireless daemon manages Wi‑Fi with NetworkManager and falls back to its **own access
point** when it cannot join a known network (IP `10.42.0.1`). So with no internet around:

1. Do not add your phone's hotspot to the robot. On boot it finds nothing and raises its AP
   (SSID/password are the ones you set during onboarding; you can also force it with
   `POST http://reachy-mini.local:8000/api/wifi/setup_hotspot` while still on a shared network).
2. Join that network from your phone.
3. Open `http://10.42.0.1:8042` for the pet's page and `http://10.42.0.1:8000` for the
   daemon dashboard (speaker volume, app logs, restart).

## Reachy's Mind (the page on port 8042)

Polls the app twice a second. Tabs:

- **Mind** – state, energy/social bars, countdowns (lonely in…, sleep in…, next reaction,
  listening-for-trick window), a running **thought stream** in plain language ("someone said
  my name from the left! listening for a trick for 8 s", "it's person #4 (friend, visit 3),
  greeting them"), and the last actions.
- **Senses** – what the eyes, ears and body report right now, plus the belly-scratch
  tuning readout with a live onset-ratio slider.
- **People** – one card per person with the 112 px face crop it enrolled from (stored only on
  the robot, under `~/.local/share/festival_pet/faces/`), visits, attention, pets, holds,
  affection. Tap two cards and **merge** them when it split one person into two entries
  (the first tapped is kept; embeddings and stats combine). **Forget** a single person, or
  everyone.
- **Controls** – wake/sleep, mute, groove intensity, face-match strictness, and a puppet
  panel: every sound, gesture and library move as a tap-to-fire button.
- **Log** – the app's own log ring (last 400 lines). The daemon's per-app log is on the
  dashboard too.

API: `GET /api/mind`, `GET /api/log?n=`, `GET /api/catalog`, `POST /api/control {cmd, value}`,
`GET /api/people/{id}/face.jpg`, `DELETE /api/people/{id}`, `POST /api/people/merge {keep, other}`.

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
pytest                                   # 29 tests, no robot needed
VOSK_MODEL=/path/to/vosk-model-small-en-us-0.15 pytest tests/test_hearing.py
```

The behavior engine takes a fake clock, so you can script whole interactions in tests
(see `tests/test_behavior.py`). `Vision.process_frame` can be driven with any BGR frame.

### Virtual robot

The SDK ships its own simulator: the daemon has a MuJoCo backend. In one terminal:

```bash
pip install "reachy-mini[mujoco]"
MUJOCO_GL=disable reachy-mini-daemon --sim --headless --no-wake-up-on-start   # drop --headless to watch it
```

then run the scripted scenario, which moves the simulated head for real while injecting a
stranger, a peekaboo, "Reachy… dance… dance", a 112 BPM track, a belly scratch, an antenna
touch, a pickup, a shake and a set-down, and checks that each reaction fires:

```bash
python scripts/sim_harness.py --vosk /path/to/vosk-model-small-en-us-0.15
```

It prints the state transitions, every action, the tempo it locked onto, and the gaze error
between the simulated head and the injected face (≈1°). `Pet` in `main.py` takes a
`RobotIO`, so the harness swaps in fake senses without touching the control loop.

## Credits and sources

- Reachy Mini SDK by Pollen Robotics (Apache-2.0): daemon, `look_at_image_pose`, recorded moves.
- Face models from the OpenCV Zoo (Apache-2.0): YuNet (`face_detection_yunet_2023mar.onnx`) and
  SFace (`face_recognition_sface_2021dec.onnx`).
- Behaviour ideas borrowed from community apps (desk_pet_bird, reachy_baby_yoda, recognizer,
  Reachy-companion): energy/social drives, priority-preempting gestures, SFace-every-N-frames,
  DoA startle.
