# Working on Festival Pet, from the robot

You are probably reading this in a Claude Code session running **on the Reachy Mini itself** (the Dev
tab in the pet's web page, or over SSH). That means you can stop guessing: the robot is right here,
the app is running, and you can watch what it sees and make it move.

## Look before you theorise

The app serves its API on `http://127.0.0.1:8042`. `scripts/petctl.py` wraps it:

```bash
python3 scripts/petctl.py see            # one line: state, face, arms, gaze, remembered spots, vision timing
python3 scripts/petctl.py watch          # ...once a second, while somebody stands in front of it
python3 scripts/petctl.py camera shot.jpg   # what the camera sees, with the detector overlays drawn on
python3 scripts/petctl.py do gesture wave   # any control the page has
python3 scripts/petctl.py controls       # every control name and its current value
python3 scripts/petctl.py trace out.jsonl --last 120   # the black box: ten rows a second of everything
python3 scripts/petctl.py log -n 80
```

**Read the camera image.** You can open `shot.jpg` directly — it has the face box, the five landmarks,
the head-pose numbers, the torso box and the arm skeleton drawn on it. Most "it can't see me" problems
are visible in one frame: the face at the edge, no torso box, the arms not tracked.

**Read the trace.** `trace.jsonl` is one JSON object per line: a header, then rows in time order.
Sampled rows (`"k":"s"`) carry what it could see, what the brain decided, where the head actually
went, the remembered places people stood, what it has blacklisted, and the state of every routine.
Events (`"k":"think"`, `"action"`, `"mark"`) are interleaved at full resolution. See the README's
"black box" section for the fields. A mark is written by `petctl mark "..."`, or the button on the
page, so ask whoever is standing there to hit it when the robot misbehaves.

## Ground rules for changes

- **Tests first, on the bench.** `python -m pytest tests -q` runs everything with a fake robot, no
  SDK needed, in about 30 seconds. Nothing gets pushed without it green.
- **Never invent defaults to paper over a failure.** If something is missing or wrong, let it raise.
  `json.loads(x) or []` and friends turn a bug into a silent shrug; the repo is written that way on
  purpose and it stays that way.
- **Prefer making an existing function do more** over adding a parallel one beside it. The composer,
  the behavior state machine and the sound engine each have one way of doing their job; a second way
  is how they rot.
- **Comments say why, not what.** Every non-obvious number in here has a reason (the speaker carries
  nothing under 300 Hz; the head hits the body frame past 30° on the body; a bracelet only sheds when
  the head is tilted). Write those down or the next change undoes them.
- **No model names, keys or secrets in anything committed.**

## The shape of it

| file | what it owns |
|---|---|
| `main.py` | the 50 Hz loop, the web API, and everything that has to touch the real robot |
| `behavior.py` | the state machine, the drives, what it remembers about people and places |
| `drives.py` | which activity it picks, scored rather than an if-chain |
| `motion.py` | every pose and gesture; `sample()` is the only thing that decides where the head goes |
| `vision.py` | the camera thread: face detection, recognition, torso, and the preview overlay |
| `pose.py` | arms from the pose model, the PLUR handshake, and the poses it has been taught |
| `sounds.py` / `songs.py` | everything it says and sings, synthesised from scratch |
| `trace.py` | the black box |
| `devconsole.py` | the terminal you are probably sitting in |

After changing anything under `festival_pet/`, restart the app from the dashboard (or kill and rerun
it) — the control loop does not reload itself.

## What it cannot do

There is no network at the festival. Claude Code needs one, so this console is a bench tool: use it
to find and fix things at home, then the robot runs the result offline for four days on its own.
