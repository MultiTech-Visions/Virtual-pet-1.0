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
| Faces | camera, YuNet at 320 px (~8 Hz) | locks onto the closest face, looks at it, micro-reacts every few seconds. The **body turns** to follow when the head is more than 12° off-centre, so the whole robot ends up facing you and the head keeps room to move. Lost faces get a real search: a widening sweep that dips down (kids, squatters) and up |
| Bodies | MediaPipe person detector (OpenCV Zoo), only when no face is visible, ~2.5 Hz | a torso in view makes it look up to where the head should be, then the face detector takes over. Toggle on the Controls tab |
| Speech | second, free-vocabulary Vosk pass on the same audio | reacts to the gist of what people say: praise → happy bounce, "cute" → shy, greetings, goodbyes → sad droop, questions → curious tilt, laughter → giggle, scolding → droop, "photo/selfie" → ta-da pose. Every sentence and what it made of it shows on the Mind tab |
| Who is it | SFace embeddings, only on new tracks / every 4 s | stranger → curious "oh? hi!", friend → happy trill + wiggle, bestie → fanfare + a library move |
| Picked up (off by default) | IMU accel/gyro, which sits in the **head** | startle, then purrs and snuggles; shaking → dizzy wobble. The IMU is ignored while the pet moves itself; enable on the Controls tab once the Senses tab shows it quiet on a desk |
| Head pets | hand rubbing the head, heard by the mics inside it | purrs and leans into the hand; keeps purring while it lasts; counts as affection for the person in front |
| Ear tickles | an antenna pushed off its commanded angle | flicks that antenna away and ducks, like a dog with its ear touched; giggles; the fourth tickle in a row gets an annoyed huff |
| A voice | mic array direction-of-arrival (4 mics in the head) | when not busy with a face, perks up and turns toward whoever started talking (at most every 4 s); its name always turns it, even mid-conversation with someone else |
| Its name | Vosk keyword spotting, only while the mic array flags speech | "Reachy!" → "huh? me?" chirp, perks up, turns toward the voice, listens for a trick for 8 s |
| Tricks | same grammar: `dance`, `hello`/`hi`, `good`, `sleep` | "Reachy, dance" → 6 s little groove; a second "dance" within 15 s → a lively library dance |
| Belly scratch | fingernail clicks on the shell, heard by the mics | ticklish giggle + wiggle, counts as a pet |
| Music | beat tracker on the mic stream (spectral flux + autocorrelation) | subtle head bob and antenna sway on the beat, three groove styles, the odd "sing-along" blip |
| Peekaboo | face hidden 0.6–3.5 s then back | giggle + bounce |
| Being stared at | same face very close for 14 s | goes shy: looks away, antennas fold, peeks back |
| Empathy | the person's head tilt (eye line) | slowly mirrors the tilt; nods back 2–4 times when you nod, shakes back when you shake |
| Mirror game | a face filling the frame (≥ 8 %) and holding for 2 s | goes quiet and copies your head pose (yaw, pitch, roll estimated from the five face landmarks), mirror-image by default (flip on the Controls tab); ends when you back away or after a minute |
| Dancing (seen) | the tracked face/body bobbing rhythmically at 50–150 BPM for 2.5 s, measured in the world frame so the pet's own bobbing does not pollute it | dances along at the tempo it sees; no microphone needed. Once locked it keeps dancing for 8 bars (min 16 s) after the rhythm was last confirmed; mid-dance there are no micro-reactions, nod-backs or lost-face searches for that person. The tempo it sees is handed to the tap clock below. **How to dance for it:** face it, stay in frame, bob your head (or shoulders, so the head goes with them) up and down a few centimetres on a steady beat for ~4 s. Hands are invisible to it |
| Tapped beat | the **Groove** card on the Controls tab | tap the beat (or the space bar) and tap "1" on the first beat of a bar; with *Manual groove* on it dances to that clock, ignoring what it hears or sees, accenting beat 1 and changing style only at 4-bar phrase turns. Grooving is all it does then: no Simon says, song, mirror game or nod-copying starts, and switching manual groove on cuts a song or game that is already running. Dials for head bob, head sway, body sway and antennas shape every groove |
| Time | – | energy drains while awake, refills asleep; lonely after 90 s alone, nods off, sleeps after 7 min; rare sneezes and hiccups |
| Sleep | – | nests its head using the SDK's sleep pose, then **motors off** and **camera paused**. The ears stay on: its name, a loud voice after quiet, a head pet or an ear tickle wake it (motors on, head lifts) |

Relationship memory is a JSON file of anonymous face embeddings plus stats (visits,
attention seconds, pets, holds, affection). Two visits apart by 2 minutes makes a
"friend"; four visits or lots of cuddling makes a "bestie". Nobody's name or photo is stored.

All sounds are synthesised in `festival_pet/sounds.py` (chirps, warbles, purrs) with
random variation so it never repeats itself exactly. Motion is layered in
`festival_pet/motion.py`: breathing + gaze + short cartoon gestures, plus optional
full-body moves from Pollen's emotions library for big moments.

### Notes on the harder senses

- **Listening** runs whenever the mic array flags speech *or* the level is clearly above the
  tracked noise floor, so a stale speech flag can never leave it deaf.
- **Name**: "reachy" is not in Vosk's English lexicon, so the grammar spots in-vocabulary
  sound-alikes ("ricci", "richie", "reach it"…) that fire on the spoken name, with decoys
  ("peachy", "reach", "beach") to absorb near-misses. Verified on synthesized speech in
  several voices (`tests/test_hearing.py`). Recognition only runs while the ReSpeaker flags
  speech, so it costs nothing while music plays.
- **Head pets**: the four mics sit in the head, so a hand rubbing it produces loud, flat
  (noise-like), continuous handling noise, unlike music (harmonic) or a scratch (sparse
  clicks). Tuning readout and slider on the Senses tab.
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
dashboard/      the port-8000 web dashboard Pollen removed in reachy-mini 1.9.0, re-mounted onto the daemon
scripts/setup_offline.sh   one-time install + model/move-library download on the robot
scripts/restore_dashboard.sh   puts the web dashboard back on http://reachy-mini.local:8000
scripts/sim_harness.py     drives the whole pet against the SDK's MuJoCo simulator
tests/                     pytest suite for everything that does not need the robot
```

## Installing and updating: the double-click installer

`installer/` holds a small desktop app that does everything below for you. Download
**FestivalPetInstaller** for your computer from the GitHub Actions "Build installer" run
(Artifacts section) of this branch, open it, and press *Install / Update*:

1. It asks for the robot's address (`reachy-mini.local`), the SSH user and password
   (factory default `pollen` / `root`).
2. It checks it can reach the robot, that the daemon is the wireless version, and that the
   robot has internet (needed the first time for the ~80 MB of models; later updates work
   offline once the models are cached).
3. It uploads the app files bundled inside the installer (or, if you untick that box, the
   latest from GitHub), runs the setup script on the robot, streams its output into the
   window, puts the web dashboard back on port 8000 (restarting the daemon), then makes the
   pet the start-up app and starts it.

Run it again whenever there is a new version: the same steps upgrade in place. On macOS the
first launch may need right-click → Open (unsigned app). Developers can also run it from
source with `python installer/reachy_installer.py` (needs `paramiko`), or headless with
`--cli`.

## Manual setup (terminal, robot online)

From your laptop, on the same Wi-Fi as the robot:

```bash
scp -r . pollen@reachy-mini.local:/home/pollen/festival_pet
ssh pollen@reachy-mini.local 'bash /home/pollen/festival_pet/scripts/setup_offline.sh'
```

The script installs the app into the daemon's apps venv (`/venvs/apps_venv`), downloads
the four OpenCV Zoo ONNX models (~57 MB, pinned by SHA-256 so they are fetched once and
only again if the pin changes) and the Vosk small English model (~40 MB) into
`~/.local/share/festival_pet/models/`, caches the
`pollen-robotics/reachy-mini-emotions-library` dataset, and runs a load check.

Then open the dashboard at `http://reachy-mini.local:8000`, find **festival_pet** in the
installed apps and start it. The pet's own page ("Reachy's Mind") is at
`http://reachy-mini.local:8042`.

### The web dashboard on port 8000 (removed by Pollen, put back here)

reachy-mini 1.9.0 deleted the daemon's web dashboard: `http://reachy-mini.local:8000` now
shows "Web Dashboard Deprecated, download the Reachy Mini Control app". The REST API behind
it is unchanged, so `dashboard/` ships the last dashboard Pollen released (1.8.4, minus its
"deprecated soon" banner) as a package, `reachy_dashboard`, that runs the same daemon with
the dashboard mounted back on. The installer does this on every run; by hand, on the robot:

```bash
sudo bash /home/pollen/festival_pet/scripts/restore_dashboard.sh
```

It installs the package into the daemon's venv (`/venvs/mini_daemon`, no dependency
changes), rewrites the daemon's `launcher.sh` so it runs `python -u -m reachy_dashboard`
instead of `python -u -m reachy_mini.daemon.app.main` (same arguments, same daemon), and
restarts the daemon. Apps, app store, move player, volume, Wi‑Fi (`/settings`), daemon
update, logs (`/logs`) all work as before. A daemon update from the dashboard rewrites
`launcher.sh`, so run the script (or the installer) again afterwards. To undo: reverse the
`-m` edit in `launcher.sh` and restart the daemon.

## Play tab: Simon says (head and arms), dance-along, singing

**Simon says, head** ("do what I do", the close-up game): start it from the Play tab (or a keypad
key mapped to `mime`). It plays a three-note fanfare, then shows a head move (look left/right/up/down,
tilt left/right; 3–5 of them, random every game, never the same twice in a row), returns to your
face and watches your head for 4 s. Copy it as in a mirror (flip with the mirror-game direction
switch) and it chirps "yes", perks, stores a fresh view of your face, and shows the next one.
Ignore it and it shows the move again, bigger, with a huff; ignore that and it shows it a third
time with an annoyed shake; ignore that and it droops, sulks and gives up. Do the whole set and it
does a ta-da. If it loses your face for 5 s it looks around, confused, and stops. Up and down are
left out when it is looking steeply up or down at you (the camera cannot read them from there).

**Simon says, arms** (flag signals): the antennas are its arms, read literally: laid back = arm
down, horizontal in front = arm out, straight up = arm up. It shows one flag position, you copy it
(as in a mirror), then it shows that one and a second, you copy both in order, and so on up to
five, like the old Simon toy. Your arms are read by the MediaPipe pose model behind the person
detector (`festival_pet/pose.py`): shoulder, elbow and wrist, as an angle from hanging down, in
three levels (down under 50°, out, up over 130°). When it cannot see your arms (too close: a face
that fills the frame has no arms in it, or the pose model is unsure) it plays the head game
instead. The Play tab shows what it reads of your arms and what it is waiting for.

**Dance-along**: while there is a beat (someone seen dancing, music heard, or the tapped clock
with manual groove on) and your arms can be read, the antennas copy your arms live, as in a
mirror, fast enough for 120 bpm (the pose runs on every camera frame then, and the close-up face
work is skipped). Every eight beats it takes two beats for a riff of its own (alternating, pumping
or a wave), then goes back to copying. Off during Simon says, library moves and sleep; switch it
off on the Play tab.

**Waves and hugs** (from the same arm reading, whenever no game or dance-along is using the
arms): wave a hand above shoulder height, three swings side to side within two seconds, and it
waves back with the mirrored antenna (your right hand, its left), tipping its head that way. Hold
both arms out wide at it for 2.5 s and that is a hug: antennas open wide, head lowered and turned
aside to nuzzle in, body rocking about five degrees, with a warm coo. A raised or open arm makes
the pose model read every frame for a moment (a wave cannot be told from a stretch at the idle
rate); the Play tab's "your arms" row shows "watching" then. One wave back per six seconds, one
hug per twenty.

**Jingles**: now and then while it is pottering about (looking around, watching someone, hanging
out) it hums a little tune it just made up: three to five bright, near-pure console blips on a
pentatonic scale, hard attack and quick decay, landing on a grid so they come out as a phrase
rather than as beeping. Two thirds of the time there is an answering phrase that repeats the
rhythm a step or two away and resolves onto the root. In the spirit of the console beeps that
answer Data's "life forms" song. Every 25–70 s, never mid-performance, mid-dance, while being
held, or when it is flat out. No switch: it is just something it does. The Jingle button on the
Play tab plays one on demand, and "jingle" is in the lexicon.

**Songs** come in two kinds, and it picks one when it decides to sing (or you pick with the
Drumline / Bass buttons). *Drumline* is rhythm, not melody: two "hands" on two pitches playing
quarters, eighths, triplets, paradiddles, flams, rolls and rests through an AABA-ish form, with a
roll to finish.

*Bass* is built the way the genre is, because without that there is nothing to follow. 140 bpm with
a sparse halftime kit underneath — kick on the 1, snare on the 3, hats on the offbeats, over a
two-bar loop, in one of three kits — so there is always a backbeat to count against while the bass
does the strange part. On top of it, a held note whose filter opens and shuts a fixed number of
times per beat (one, two or three, always locked to the grid), or wahs, lasers or offbeat stabs.

Arrangements are built from four-bar phrases by a small grammar, so no two songs have the same
shape — four days of festival is a lot of songs to sit through. The grammar is what keeps them
followable: every song counts you in with a bar or two of kit alone, every drop has a build in
front of it and lands on a phrase line, a breakdown always builds back into another drop, and every
song has an ending. Within that it picks its own phrases (count-in, drop, ride, breakdown, outro,
each in two variants), how many, which three bass voices fill the A/B/C slots, and a riff that moves
the bass note around under each phrase. Songs run 21 to 41 seconds, some with one drop and some with
two.

The speaker reproduces nothing under about 300 Hz, so the bass note sits at 310–370 Hz and the
harmonics with the moving filter do the talking; an actual sub would come out as silence. The kick
is a 440→150 Hz drop with a click on the front, and the click is what carries it. The body bobs at
half the song tempo, on the 1 and the 3, which is both the halftime feel and as fast as the neck
wants to move. Saved songs from before the two kinds existed are stamped as drumline when they load.

**Performing**: a song on its own is just a song — it sings its little song and is pleased with
itself, a wiggle and a happy beep. But it watches the audience while it plays: a face in view with
their head pointed at it counts as watching, and if more than half the song was watched it lines up
another one after a short pause ("they're still watching! one more"). After the second, a third is
a one-in-three rarity, so it stays special. Only a set of two or three earns the full house bow —
turn 30° right, bow, 30° left, bow, centre, bow — which means that when you see the big routine, it
means something. The Mind tab shows the set: which song it is on, what fraction of it is being
watched, and when the encore lands.

**Trading kandi (PLUR)**: at a festival, people will want to trade bracelets with it, so it knows
the handshake. Finger poses are past what the pose model can give at across-the-tent distance, so
it reads the four steps from arm positions instead, and the sequence is what makes it reliable:
each step only counts after the one before, held half a second, with twelve seconds to get to the
next one or the handshake lapses.

| Step | You | It |
|---|---|---|
| **Peace** | both arms up at 45°, hands well apart and above your shoulders | antennas snap up into a V, excited chirp |
| **Love** | hands together up at your chest, making a heart | antennas arc inward until the tips nearly meet, a coo |
| **Unity** | hands clasped and lowered right down in front, arms in a V | antennas fold in, it snuggles, content |
| **Respect** | one arm up and bent, forearm and fist straight up at head height, other arm down | it holds out an antenna |

A hug is the same hands-apart shape as peace but with the arms straight out at shoulder height, so
the height of the wrists separates them; while a handshake is under way the wave and hug detectors
stand down anyway.

**The exchange.** On *respect* it trades, both ways. It is wearing bracelets on its antennas (tell
it which ears are loaded on the Controls tab), so first it gives one:

1. It raises the loaded ear and looks at you.
2. It rolls its head over until that ear's base is the lowest part of the head.
3. It lowers that antenna, slowly, until the bracelet runs off the tip into your hand, and giggles.
4. It holds there a moment while you take it, then comes back up level.

Then it asks for one back on the same ear: that antenna goes to just past vertical and tipped
toward you, the other leans out of the way, and it **freezes** — no breathing, no groove, no
gestures — so you can slide one down the wire and let gravity take it to the head. It knows the
bracelet landed because the antenna gets pushed off its commanded angle, which is the same detector
that feels an ear tickle; that touch is swallowed rather than passed to the brain, or it would
flinch at exactly the wrong moment. It waits ten seconds in case you are digging one out of a bag,
then gives up gently. Two seconds after one lands it eases the antenna and head back to normal.

Wearing one is not a special mode: a **loaded antenna is simply held within 26° of vertical** for as
long as it is loaded, whatever else the robot is doing, which is all a bracelet needs to stay on
through a whole dance. Only the twenty seconds right after a trade also take the size out of the
head's movement, to settle. Trades are counted per person and are worth a lot of affection, so a
trader becomes a bestie fast.

Both the give and the ask are driven through overrides that already existed — `hold` for the head
pose (Simon says shows poses with it) and `show_arms` for absolute antenna angles (the arm game) —
so there is no third way of moving the head to keep in step with the rest.

The **Kandi trading** card on the Controls tab has: trade now (or from a named ear), just ask for
one without giving, cancel a trade that started by mistake, a per-ear loaded toggle, and a **shed
tilt** slider. Which way the head has to lean to make a given ear the low point is a fact about the
real robot, so it is a signed number you can set from the page: if it tilts the wrong ear down, use
a negative value.

**Lexicon**: every sound with what it means**Lexicon**: every sound with what it means, tap to hear; it lives on the Controls tab under
"Puppet it". Distinct calls: a rising two-note for "let's play mirror" (falling for "mirror
over"), the fanfare for Simon says, a double blip before each shown move, a bright "yes", a
puffed "huff".

**Sneeze**: 10.6 s. It stops looking at you, looks down with the antennas laid right back, gives
a little shake, lifts three times with rising inhales while the antennas climb a step each time,
whips them up and crossed with a squeak, then CHOO: head down hard, antennas out wide; slow
recovery with a droop, then a clearing shake. If you are smiling at it afterwards, it giggles.

## A keypad in someone's hand

Any Bluetooth (or USB) keyboard the robot is paired with is a hand controller: the app reads
`/dev/input` directly, so a key press reaches the pet in a millisecond with the kernel's own
timestamp, and a tapped beat is not smeared by radio latency. The four-key PCsensor MK424 is
the intended one (sends A B C D, PIN 1234, its "S" button is its own mode key and is ignored);
any keyboard works for trying it out.

Controls tab, **Keypad** card:

- *Scan for keyboards* lists what is discoverable; *pair* pairs, trusts (auto-reconnect) and
  connects with the PIN in the box. Paired devices are listed with a *forget* button. Needs
  `bluetoothctl` on the robot (`apt install bluez` while online if the card says it is missing).
- Layers: the MK424 has three layers (its LED colour shows which is on), four keys each; the
  pet uses the first two and ignores the third. One tap = one action (set the pad not to
  auto-repeat); there are no hold actions. The actions are fixed per layer; type the key code
  each key sends (press it, read "last key").
  1. **Dancing**: groove left · tap the beat · tap the "1" · groove right. Any key on this
     layer turns manual groove on. Left / right say which way to groove, they do not point it:
     it leans that way (head roll and yaw, one antenna forward and one back, about 5° of body)
     and keeps dancing with whoever it is with, fading over a couple of seconds. Keep tapping
     the same way and the lean grows to full and it tilts its head over as well (at most every
     2 s). The body never turns away from the person. Left-right-left-right within a second (a
     fighting-game combo) stops the dancing: manual groove off, the tempo cleared. A tempo
     tapped faster than 150 bpm grooves at halftime (every other beat, on the "1" and "3" once
     the "1" is known): the daemon does not smooth the 50 Hz targets, and a full bob every
     0.4 s rattles.
  2. **Petting** (one press of the pad's mode key from dancing lands here): head pat · chin
     scratch · ear rub · belly rub. A press does not fire an animation. It keeps a hand on the
     robot for a moment (the same continuous fold-and-lean a real rub gets) and tops up a
     build-up, so drumming on all four keys like a fidget toy reads as one long cuddle rather
     than a fit of gestures. The build-up passes four marks, each once per session and never
     closer together than 2.5 s: a curious perk, a contented lean, a purring snuggle, and
     melted. It ebbs away over about 6 s once you stop, and a gap of 3 s starts a fresh
     session. Every press counts toward being its friend (at most one every 2 s).
- Any key counts as interaction, so the pet does not get lonely while someone plays with it.

The restore script adds the `pollen` user to the `input` and `bluetooth` groups (needed to read
keyboards and to pair); that takes effect when the daemon restarts, which the script does.

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
- **Senses** – what the eyes, ears and body report right now (fixed rows, nothing jumps), a live 20 s chart of where the person is (world yaw/pitch, the dance amplitude floor and when it thinks they are dancing), a live 20 s microphone chart
  (loudness, head-rub energy, belly-scratch band energy, flatness, with the current floors
  drawn as dashed lines) and one-tap **Calibrate** buttons for head pets and belly scratches:
  stay quiet 3 s, touch for 3 s, and the thresholds are set from what it heard (it refuses
  if the touch was not clearly louder). Manual sliders for ratio, floor and flatness too.
- **People** – one card per person with the 112 px face crop it enrolled from (stored only on
  the robot, under `~/.local/share/festival_pet/faces/`), visits, attention, pets, holds,
  affection. Tap two cards and **merge** them when it split one person into two entries
  (the first tapped is kept; embeddings and stats combine). **Forget** a single person, or
  everyone.
- **Controls** – wake/sleep, mute, pickup on/off, body finder on/off, **ears on/off** (stops all
  microphone processing, for dead mics or CPU; also hides the microphone cards and "what it heard"), mirror-game direction, groove intensity, a head-forward slider (slides the head
  forward by up to N mm as it looks up, so the back of the head clears the body; default 12),
  face-match strictness, speaker volume, shut down / reboot, a **Groove** card (manual groove
  on/off, *Tap beat* / *Tap "1"* buttons with a live bpm / beat / bar readout, a BPM box, and
  head-bob / head-sway / body-sway / antenna dials), and a puppet panel: every sound, gesture
  and library move as a tap-to-fire button. Library moves are recorded body-forward; they are
  turned with the body's current heading when played, so they work while it faces you sideways.
  All switches and sliders are remembered across restarts (`~/.local/share/festival_pet/settings.json`).
- **Log** – the app's own log ring (last 400 lines). The daemon's per-app log is on the
  dashboard too.

With ears off, every microphone card on Senses is hidden.

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

## Which version is on the robot?

The Mind page header shows `vX.Y.Z`; the first line of the app log (Log tab) adds the commit
and upload time. The version is `pyproject.toml`'s and is bumped with every change that is
pushed; the commit and upload time come from a stamp the installer writes at upload time, so
two uploads of the same version are still told apart. The installer's own steps print the
version it is sending and `v<before> → v<after>` after installing.

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
