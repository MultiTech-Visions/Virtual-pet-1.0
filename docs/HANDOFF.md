# Where this is up to

Written for a Claude Code session picking the work up on the robot itself. `CLAUDE.md` says how to
drive the pet (`scripts/petctl.py`, the trace, the camera overlay); this says what is broken right
now and what was just changed, because none of that is in the code.

Last known good: **v0.16.3**, 209 tests passing, branch `claude/sweet-brahmagupta-wflvnz`.

## Open, reported from the field, not yet fixed

Both came in together and neither has been diagnosed. Start here.

1. **"Tap to stop doesn't work."** The kandi walk-through (Play tab → the manual PLUR buttons) has a
   stop that does not stop it. `Pet.stop_plur` (`main.py`) clears `signs.plur_step` and
   `_plur_hold_until` and calls `composer.ears_clear()` — but it does *not* touch `_kandi_offer_until`
   or `_kandi_give_t0`, so if the walk-through has already reached the trade, stopping the handshake
   leaves the trade running. Check what the page's stop button actually posts, and whether it should
   be calling `cancel_kandi` as well. Do not paper over it with a "stop everything" flag; find which
   clock is still set. `petctl.py do plur_next` / `do stop_plur` drive this from the shell, and
   `/api/mind` reports `kandi.offering`, `kandi.side` and `plur.step` live.

2. **It crashed when asked to offer the RIGHT ear.** Left works. Right (side index 1) crashed the
   app. No traceback was captured. Reproduce with `python3 scripts/petctl.py do trade_kandi right`
   and `do start_kandi right`, then read `petctl.py log -n 120` — there is a global exception handler
   installed in `install_routes` that logs the traceback into LOG_RING, so it should be there. The
   bench tests cover both sides of `start_kandi` and pass, so it is likely something only the real
   robot path touches (the SDK call, `show_arms`, or a `None` `kandi_side`).

## Sides: the thing that keeps biting

**Antenna index 0 is the robot's LEFT ear.** `SIDE_NAMES = ("left", "right")` in `main.py` is the one
source of truth; `_side_name` and `_side_arg` convert. Three separate bugs have come from code still
carrying an older "0 is right" convention:

- the trade buttons picked the wrong ear (fixed in `2cae988`),
- the offer pose turned the head toward the offered ear instead of away from it (fixed in `20023a0`),
- `motion.py` still calls the two antenna variables `ant_r` / `ant_l`, where **`ant_r` is index 0 and
  therefore the LEFT ear**. That naming is inherited from the SDK's own ordering. Do not "fix" it by
  renaming without checking every use; do read it carefully before changing any sign.

The rule the offer pose follows: present an ear → raise that antenna, **turn the head away from it**
(negative yaw for the left ear, positive for the right), and tilt toward it. Turning away is what
swings the ear round to face whoever is standing there.

## Recently changed, in case something regressed

- `20023a0` — offer/give head yaw flipped. Test:
  `test_the_offered_ear_is_turned_to_face_the_person_and_the_other_stays_safe`, now parametrized over
  both sides.
- `2cae988` — unity is **arms folded across the chest** now, not clasped hands, measured by the new
  `cross` feature in `pose.py`. **Unity needs retraining on this robot**; the old prototype is for the
  old gesture. `hug` is trainable too, and training keeps the median of the last 5 goes
  (`PLUR_TRAIN_KEEP`).
- `2cae988` — antenna "holds" (the PLUR countdown ear) are eased and rate limited to
  `EAR_HOLD_SPEED = 4.0` rad/s after the real one was measured flicking at 13 rad/s and startling
  people.
- `d06a827` — the Dev console now prints the exit status of a dead session instead of closing
  silently, and installs Claude Code with `sudo -i` (npm's global prefix is root-owned).

## House rules that are not negotiable

From `CLAUDE.md`, repeated because they get broken first:

- `python -m pytest tests -q` green before anything is pushed. ~55 s, fake robot, no SDK.
- **No invented defaults.** No `json.loads(x) or []`, no swallowed exceptions, no silent fallbacks.
  If something is wrong it must raise. Three of the bugs above were invisible precisely because
  something failed quietly.
- Prefer making an existing function take another argument over adding a parallel one beside it.
- Comments say *why*. Every odd constant here has a reason; write it down or the next change undoes it.
- No model names, keys or secrets in anything committed.
- Restart the app from the dashboard after changing anything under `festival_pet/` — the control loop
  does not reload itself.

## Context that is not in the code

- This runs at a campsite for four days with **no network**. The Dev tab and Claude Code are bench
  tools only; the app itself must stay fully offline (`HF_HUB_OFFLINE=1`, models on disk, all audio
  synthesised). If the festival network is gone the robot raises its own AP at `10.42.0.1`, but only
  on boot — pull the network then reboot if you want to test that.
- The person working on this is autistic and asked for the controls to be ordered and predictable:
  buttons in rows of two, consistent left/right, no surprises. Keep that.
- Bump `version` in `pyproject.toml` on every change that gets pushed.
