# Skatepark experiment

## Run it

```bash
# Inspect the board/park physics (constant stance, not a learned policy).
uv run scripts/play_skatepark.py --preview

# Watch a downloaded policy. Difficulty is 0..1; 0 is flat.
uv run scripts/play_skatepark.py --checkpoint logs/modal/<run>/model_500.pt \
  --course rollers --difficulty 0.6

# Browser viewer with velocity controls and a "Next skateboard request" selector.
uv run scripts/play_skatepark.py --checkpoint logs/modal/<run>/model_500.pt \
  --viewer viser --tricks chain

# Evaluate first episodes without resets hiding falls; play never uses assisted spawns.
uv run scripts/play_skatepark.py --checkpoint logs/modal/<run>/model_500.pt \
  --headless --num-envs 64 --seconds 12 --difficulty 0.6 --tricks ride

# Record duck zero to MP4 (also reports evaluation metrics).
uv run scripts/play_skatepark.py --checkpoint logs/modal/<run>/model_500.pt \
  --video logs/skatepark.mp4 --seconds 12

# Smoke before a long run, then submit to a B200 and disconnect.
modal run scripts/modal/train.py --wait --task Mjlab-Skatepark-MicroDuck \
  --num-envs 64 --iterations 5 --name skate-smoke
modal run --detach scripts/modal/train.py --task Mjlab-Skatepark-MicroDuck \
  --iterations 10000 --name skatepark
```

Available courses: `cruise`, `banks`, `rollers`, `tabletop`, `quarterpipe`.
The quarterpipe preset is a shallow curved transition, not a vertical wall.
There are no gaps or grind rails in this first curriculum. `--tricks` accepts
`auto`, `ride`, `ollie`, `180`, or `chain` (alternating ollie and 180 requests).
Forced requests are evaluation controls, not evidence a checkpoint has learned
those tricks. The GUI applies a selection at the next request boundary.
`--speed` and `--turn-rate` fix navigation commands. Use `--start-speed 0 --speed 0.4`
on a flat course to distinguish self-propulsion from coasting with initial momentum.
Headless reports include measured forward speed and yaw rate.

See [Modal operations](modal.md) for downloading, stopping and resuming runs.

## Physical and observation contract

The skateboard is an independent free body with four passive wheel hinges and
two sprung, inclined steering kingpins. Only the robot's 14 BAM servos are
actuated. Feet interact with the deck through frictional contacts; there are no
welds, hidden thrusters, or prescribed board trajectories. Initial rolling
velocity is applied coherently to duck, deck and wheels once at reset.
After the riding curriculum, 20% of training resets start 3.5–5 cm above the
surface to teach landings. These assisted landings are counted separately and
cannot promote the trick curriculum. They are disabled in playback/evaluation.

The existing 61D proprioception/command observation group remains unchanged.
An additional simulation-only `skate` group supplies board-relative state,
terrain preview and trick goals to the actor and critic. This is a separate
skateboard-policy contract, not a hardware-hot-swappable 61D policy. The existing
publisher's shape check must reject it. Native `play` consumes both groups.
The actor input is 107 values: original 61, board/rider state 26, terrain rays 15,
and trick command 5. The latter is a ride/ollie/180 one-hot plus sine/cosine of
the remaining board-heading error. After success the visible goal returns to
ride, making repeated identical requests distinguishable. The 180 goal measures
board rotation (shove-it-style); it does not require the duck's body to turn 180°.

Terrain rays begin a meter above the board so they can see approaching uphill
surfaces. Board state includes relative rider position/yaw, board linear/angular
velocity and gravity, kingpin angles, wheel speeds and contact flags. Robot
actions are still 14 unfiltered BAM joint targets. The target offset differs
slightly from HOME to compensate static servo loading on the board; ONNX
metadata records it as `skate_action_offset` and identifies `skatepark-v1`.

An episode contains multiple requests: ride, pop, land and ride away, then another
request. Landing credit requires airborne board motion followed by wheel support
and feet on the deck; a duck jumping off its board is not a successful trick.
No board part, including the kicktails, may touch the floor during credited
airtime. Landing requires three supported wheels, both feet on the deck, upright
duck/board, small heading error, positive ride-away speed, and 0.12 s of settled
support. A 180 additionally requires airborne rotation, so turning on the ground
and then hopping does not qualify. Each request pays once; a new request needs a
fresh flight. Progress is a bounded frontier increment, not a hovering reward.
Completed requests return to riding without resetting physical state.
Command state owns trick latches and resets. Terrain promotion and trick exposure
must depend on measured competence, not only elapsed training iterations.

Largest initial training shape: 4096 independent duck/board pairs on a seeded grid
of flat, bank, roller, tabletop and curved-transition lanes. Each pair has two free roots, 14 robot
servos, and six passive skateboard joints. Resets write both entities through
their own joint/root mappings. GPU contact budgets cover terrain, deck and robot.

Evidence required before a long run: CPU model/contact checks, noisy 3 s settling
with tilt and deck-relative stance measurements, coherent rolling and lean/steer
checks, reset isolation, trick reward failure cases, and a 64-env/5-update GPU
smoke test with export. Actual trick quality remains unproven until checkpoint
rollouts show it; learning rates or reward totals alone do not establish success.

## Curriculum and quality

| Stage | Unlock | Experience |
|---|---|---|
| 0 | Initial | Flat balance, forward speed/turn commands, stationary starts |
| 1 | At least 500 updates and riding-success EMA > 70% | Competence-based terrain promotion/demotion |
| 2 | At least 1200 updates and riding-success EMA > 70% | Ollie requests plus separately counted landing practice |
| 3 | At least 2000 updates, unassisted trick EMA > 50%, and 100 successful tricks | Mixed ride, ollie and airborne 180 requests |

A riding success requires more than seven seconds aboard, enough travel for the
entry speed (or a stationary-start trial), and no fall termination.
Terrain promotion additionally requires 2.5 m of travel; idle trials cannot
advance terrain difficulty. Head/robot DR
and sensor delays come from the walking recipe. Slip and light action-rate costs
apply throughout; impact and torque-rate costs ramp with measured unassisted
trick competence, after discovery. No angular-speed cap blocks physically
necessary trick rotations.
Actual time since physical reset is used for this gate, not PPO's randomized
initial episode-length buffer. Loading a checkpoint rebases episode ages before
reset callbacks. New-episode heading targets likewise use the known reset frame,
not stale pre-forward body poses from the previous episode.

Checkpoints preserve the adaptive stage, EMAs and counters in `skate_curriculum`,
plus terrain levels in `skate_terrain_levels`. Same-sized resumes restore levels;
different-sized resumes sample their saved distribution. Actor-only playback
keeps the requested course/difficulty. Physical state and in-flight requests
restart, as with other mjlab checkpoints. Changing source/recipe for a resume is
an experiment change; each Modal run retains its actual source archive.

Useful metrics: `skate_ride_ema`, `skate_trick_ema`, `skate_speed`, `skate_chain`,
`skate_terrain_level`, and the command's requested/completed/assisted counts.
Measure headless survival and watch videos as well as reward curves. Initial
rolling momentum and downhill gravity help discovery; sustained self-propulsion
from rest and reliable trick chains are separate behavioral checks, not assumed
from staying aboard.

## Verification and first experiment

`uv run scripts/skatepark_probe.py` holds fixed targets for three seconds at
nominal BAM settings with noisy joint starts. With 16 environments per condition:

- Stationary: 16/16 aboard, maximum tilt 4.1°.
- Rolling at 0.5 m/s: 16/16 aboard, maximum tilt 6.3°, median travel 1.26 m.
- Landing-practice drop: 16/16 aboard after landing, 16 assisted landings,
  **zero** counted self-initiated tricks.
- Partial resets left other environments untouched; repeated resets did not
  accumulate offsets. All states stayed finite.

CPU tests check passive mechanics, physically mirrored steering, compiled ramp
surfaces by raycast, HOME foot height on the actual robot, reward loopholes,
request transitions, and isolation from other tasks. Local and B200 64-env /
five-update smoke runs passed; the normalized ONNX is `[1,107] → [1,14]`.
Full checkpoint resumes, including terrain-state restoration, were exercised.

The first B200 pilot is `20260916-174816-skatepark-v1-82b07b89` in
`microduck-runs`. At checkpoint 150, 64-environment / 12-second deterministic
evaluations gave 64/64 survival on flat and 20/64 on rollers at difficulty 0.6.
These demonstrate early balancing/coasting, not mastered propulsion or tricks.
The pilot reached curriculum stage 1 after iteration 500. Later source adds
landing practice, competence-gated quality costs, explicit ride-away goals,
forward-only speed tracking and stronger turn tracking;
continue the pilot checkpoint in a new run rather than overwriting its provenance.

At checkpoint 1850 the pilot still had zero recorded self-initiated tricks.
Roller-course survival improved to 64/64, but mean travel was only 1.05 m: it was
braking before the features. A stationary-start test (32 environments, forward
command 0.4 m/s) produced only 0.0019 m/s mean forward speed. Survival alone was
not skating competence. The revised recipe gates pose shaping by actual command
tracking, so parking cannot retain most of the positive reward stack. Its longer
continuation starts from the earlier balanced-rider checkpoint 500 rather than
the later parking policy. Tricks, propulsion and chains remain evaluation targets.
