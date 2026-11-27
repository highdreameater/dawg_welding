# Slide-by-slide explanation (presenter notes)

Covers every slide in `slides.tex` except slide 13 ("Why not the simpler versions?").
For each slide: the idea in one line, what every symbol means and why it is there, and how to read each figure.

---

## 1. Title: Autonomous Force-Controlled Surface Treatment with a Legged Mobile Manipulator

**Idea:** a quadruped (Unitree Go2) with an arm (Z1) finds a job, walks to it, and presses a tool on a curved surface with a controlled force.

**Image (`final_vessel.jpg`):** the robot at the vessel after tracing its work order.
- The **shading on the surface** is the measured contact force at each spot. An even colour means the force stayed near 5 N.
- The **inset** is the head camera view with the ArUco marker. That marker is how the robot knows where the job is.

**Why "5 N"?** It's the target contact force for the whole project: firm enough to be real contact (sanding, coating, probing), light enough for a small arm.

---

## 2. Why does surface work need legs and an arm?

**Idea:** the job sites (tanks, pipes, hulls) are behind stairs and on uneven floors, the surfaces are curved, and the task is about contact force, not just position. Only legs plus an arm covers all of that.

**Table:** a qualitative check, not measured data.
- **stairs:** can it get there? Wheels and fixed arms can't.
- **reach:** can it put a tool on the surface? Legs alone can't.
- **force:** can it press with a controlled force? Legs alone can't (no tool).
- **mobile:** can it move between jobs? A fixed arm can't.
- Only **legs + arm** has four ticks. That's the whole argument for the platform.

**Images:** top, the Go2 climbing 10 cm stairs with the arm mounted (shows the mobility). Bottom, the arm working on a wall next to a stair platform (shows the work happens in cluttered places).

---

## 3. The task: work orders on surfaces the robot has never seen

**Symbols**
- **$W\times H$:** width × height of the rectangle to cover, given in the work order.
- **marker $k$:** the ArUco sticker with ID $k$ (5, 7, 23, 37 in the runs). The work order says "cover a $W\times H$ area next to marker $k$". The marker anchors the job on the surface.
- **$F^\ast = 5$ N:** the desired contact force. The star always means "target / reference" in these slides.
- **"along the surface normal":** the force has to push straight into the surface, perpendicular to it. On a curved surface the normal changes from point to point, so the tool has to keep re-orienting.

**Given / Unknown / Objective:** frames the problem honestly.
- The robot gets a 2-D floor map only.
- It doesn't know where the markers are or what shape the surfaces are.
- Its own position estimate drifts (odometry).

**Images:** the four test surfaces, each with one sticker.
- horizontal **pipe:** curved in one direction
- **wavy panel:** freeform shape
- sphere **tank:** curved in two directions
- **vessel:** large curved body

They're chosen so that no single shape assumption (a plane, a cylinder) would work for all of them.

---

## 4. How the robot thinks: five layers, one job pipeline

**Left stack (the layers, top = most abstract):**
1. **Localise:** where am I? Odometry gives a drifting guess, and matching the depth scan against the map corrects it (slide 9).
2. **World model:** what do I know? The map, markers found so far, surface models, and which areas have been seen.
3. **Decide:** what next? A small state machine: *work* (a known job is waiting), *look around* (scan the surroundings), *explore* (go somewhere new), or *done*.
4. **Execute:** do the job, through the six steps on the right.
5. **Control:** move. Either the RL walking policy (moving between places) or the stance WBC + arm (standing still and working).

**Right box (the six steps, identical for every surface):**
1. **approach:** walk to near the marker.
2. **survey $w(u,v)$:** take depth images and build a surface model.
   - $u,v$ are 2-D coordinates on the work-order rectangle (like a sheet laid over the surface).
   - $w(u,v)$ is the surface height (depth) above that sheet at each point.
   - This turns a 3-D surface into a simple height map.
3. **plan reach:** pick where to stand (standoff) so the arm can reach the whole area (slide 8).
4. **dock:** walk precisely to that standing spot.
5. **work 5 N:** trace the area while holding 5 N.
6. **check:** compare what was covered with what was asked.

**Red dashed arrows (the recovery paths):**
- **redo / leftover patch:** if *check* finds uncovered parts, go back to *dock* (possibly at a new standoff) and do the leftover piece.
- **re-acquire marker:** if the marker is lost during the survey, look for it again.

**Dashed frame label "same six steps for every surface":** the key design point. There's no special code per shape.

---

## 5. Walking with 30 % extra mass: RL policy and payload fine-tuning

**Symbols in the bullets**
- **$\pi(o_t)\rightarrow a_t$:** the policy $\pi$ (a neural network) maps the observation $o_t$ at time $t$ to an action $a_t$.
  - $o_t$ holds joint angles and speeds, body orientation, the velocity command, and the last action.
- **50 Hz:** the policy runs 50 times a second.
- **$q^\ast = q_0 + 0.25\,a_t$:** how the action becomes a motor target.
  - $q_0$ is the default standing joint pose.
  - $a_t$ is the network output.
  - 0.25 is the action scale, so the network only nudges joints around the standing pose.
  - $q^\ast$ is the desired joint angle, tracked by a **joint PD** controller (spring + damper on each joint).
- **MoE-CTS:** a pretrained stairs policy. MoE = mixture of experts; CTS = concurrent teacher–student training.
  - "**blind**" means it uses no camera, only the body's own sensing.
  - "**zero-shot**" means it was used as-is, not retrained.
- **PPO:** Proximal Policy Optimisation, the standard RL algorithm used for the fine-tuning.
- **warm-started actor + critic:** start from the pretrained networks instead of from scratch.
  - The actor is the policy; the critic estimates how good a state is.
  - This is why it only takes 24 minutes on a laptop CPU.
- **curriculum: arm mass ×0.6 → 1.25:** training starts with a lighter-than-real arm and slowly makes it 25 % heavier than real, so the policy learns to carry it gradually.
- **random base mass, friction, pushes:** domain randomisation, so the policy doesn't overfit to one exact robot.
- **6.5 M samples:** 6.5 million simulation steps of experience.

**Top figure (`train_curve.png`, three panels vs training iteration):**
- Grey is the raw value each iteration; blue is the smoothed trend.
- **episode return:** total reward per episode. It rises from ~0 to ~37, so the policy keeps getting better at the walking objective.
- **fall rate:** fraction of episodes that end in a fall.
  - It's 1.0 at first because the pretrained policy falls with the arm mass added.
  - It drops to under 0.1.
- **episode length [s]:** how long an episode lasts before ending. The cap is 20 s, so ~19–20 s means the robot almost always survives the full episode.

**Bottom figure (`loco_finetune.pdf`, grey = pretrained, blue = fine-tuned with arm):**
- **velocity tracking:** achieved speed ÷ commanded speed, in %. The 100 % dashed line is perfect tracking.
  - Fine-tuned is better forward (93 → 100 %), turning (82 → 84 %) and sideways (81 → 94 %).
- **body tilt with the arm:** the worst body tilt seen. It drops from 18° to 5°: the robot no longer leans under the arm.
- **falls:** out of 30 random trials, at the real arm mass and 1.25× mass.
  - Pretrained: 3 and 1 falls. Fine-tuned: 0 and 0.

**Takeaway:** falls go from 3/30 to 0/30 and tilt from 18° to 5°. The caveat is that the stairs policy was not fine-tuned with the arm.

---

## 6. Holding still while the arm works: the stance WBC as a QP

**Idea:** when the robot stands and works, one optimisation decides all joint torques (legs and arm) every 2 ms. It moves the tool where it should go while respecting physics. WBC = whole-body controller; QP = quadratic program, an optimisation with a squared-error cost and linear constraints, which solves very fast.

**"Solved every 2 ms (DAQP, ≈0.2 ms)":** it runs at 500 Hz. DAQP is the solver, and one solve takes about 0.2 ms, so there's plenty of margin.

**The cost (what we want):**
$$\min_{\ddot q,\,f}\ \sum_i w_i\big\|J_i\ddot q + \dot J_i\dot q - a_i^\ast\big\|^2$$
- **$\ddot q$:** the accelerations of all joints *and* of the floating base (6 base + 12 leg + 6 arm). This is the first decision variable.
- **$f$:** the contact forces at the four feet (3-D each). This is the second decision variable.
  - The optimiser chooses accelerations and foot forces together, so they're always consistent.
- **$i$:** the index of a *task*. The tasks are: keep the base rigid, put the tool at its pose, keep the arm in a nice posture.
- **$w_i$:** the weight of task $i$, i.e. how much we care about it. Base weight 100 means the body must stay still above all else.
- **$J_i$:** the Jacobian of task $i$. It converts joint motion into task motion (e.g. joint speeds into tool speed).
- **$J_i\ddot q + \dot J_i\dot q$:** the task acceleration that results from $\ddot q$.
  - The $\dot J_i\dot q$ term is the extra acceleration that comes from the robot already moving (the Jacobian changes as joints move).
- **$a_i^\ast$:** the acceleration we *want* for task $i$. The cost is "achieved minus wanted, squared". It's zero when all tasks are met.
- **$a_i^\ast = K_p e_i + K_d\dot e_i$:** a PD law in task space.
  - $e_i$ is the task error (e.g. how far the tool is from its target).
  - $\dot e_i$ is how fast that error is changing.
  - $K_p$ acts like a spring pulling the error to zero; $K_d$ acts like a damper that stops oscillation.

**The constraints (what must be true, no compromise):**
- **$M_b\ddot q + h_b = J_{c,b}^\top f$ (floating base):** Newton's law for the body. The base has no motor, so it can only be accelerated by the foot forces.
  - $M$ is the mass/inertia matrix, $h$ is gravity + Coriolis/centrifugal forces, and the subscript $_b$ means "the 6 base rows only".
  - $J_c$ is the contact Jacobian (feet). $J_{c,b}^\top f$ is how the foot forces push on the base.
- **$J_c\ddot q + \dot J_c\dot q = -k_c J_c\dot q$ (feet fixed):** feet should not accelerate.
  - $J_c\dot q$ is the foot velocity, and the right-hand side $-k_c\cdot$(foot velocity) is a small correction that brakes any slip back to zero.
  - $k_c$ is that correction gain. It's the "foot vel. feedback" bar in the ablation.
- **$|f_{x,y}|\le\mu f_z,\ f_z\ge5$ N (friction, μ = 0.6):** the friction cone.
  - The sideways foot force must be at most μ × the vertical force, otherwise the foot slides.
  - $f_z \ge 5$ N keeps every foot firmly loaded (no foot about to lift off).
- **$|\tau|\le\tau_{\max},\ \tau = M_a\ddot q + h_a - J_{c,a}^\top f$:** motor torque limits.
  - $_a$ means the actuated joint rows (legs + arm).
  - $\tau$ is the torque each motor would need, and it must stay within the real motor limits.

**Bullets:**
- **"tool pose (+ clamped integral)":** an integral term removes a small steady offset. It's clamped so it can't wind up.
- **Z1 joint friction (1 Nm):** the arm joints have friction that the model doesn't know about. It's compensated in one of two ways:
  - **feed-forward:** add the expected friction torque for a planned motion.
  - **disturbance observer:** estimate the unknown torque from the error between predicted and actual motion, then cancel it.

**Flowchart (right):**
- references (base, tool, posture) → **QP** gives $\ddot q^\ast, f^\ast$ → **inverse dynamics** $\tau = M\ddot q^\ast + h - J_c^\top f^\ast$ + friction compensation (turns the chosen accelerations into motor torques) → **robot** (torque-controlled motors).
- The robot's measured $q,\dot q$ (joint positions and speeds) update a **model copy** ($M, h, J, \dot J$ recomputed each step). The model feeds the next QP, which closes the loop.

**Takeaway numbers:** on the Go2 the body moves 0.4 mm and the tool tracks to 0.08 mm. Treating the physics as hard constraints is what buys this precision.

---

## 7. Pressing just right: hand-off and force control

**Hand-off chain:** walk (RL) → **settle** (all 4 feet down, body still) → WBC + arm → walk (RL).
- Settling first matters: the WBC assumes the feet are fixed, which isn't true mid-step.

**Admittance (the force loop):**
$$\dot\delta = k_a\,(F^\ast - F),\qquad p_{\text{tool}} = p_{\text{surf}}(u,v) - \delta\,n$$
- **$F^\ast$:** target force (5 N). **$F$:** measured contact force.
- **$\delta$:** how far "into" the surface the tool target is placed (a depth offset).
- **$\dot\delta$:** the rate of change of $\delta$. If the robot presses too little ($F<F^\ast$), $\delta$ grows and the target moves deeper, so it presses harder. If it presses too much, $\delta$ shrinks.
  - It's an integrator on force error, so there's no steady-state force error.
- **$k_a$:** admittance gain, i.e. how quickly $\delta$ reacts. Higher is faster but more prone to oscillation.
- **$p_{\text{surf}}(u,v)$:** the surface point at work coordinates $(u,v)$, from the survey model $w(u,v)$.
- **$n$:** the outward surface normal at that point.
  - $-\delta n$ means "$\delta$ metres inward along the normal", so the push is always perpendicular to the surface.
- **$p_{\text{tool}}$:** the resulting tool target that goes to the WBC.
- **Intuition:** the robot doesn't command a force directly. It commands a position slightly *inside* the surface, and adjusts how far inside until the force reads 5 N.

**Feed-forward $F^\ast n$:** tell the WBC in advance "the tool will feel 5 N along $n$", so the dynamics already account for it.

**Tangent plane:**
- The tool moves in a **raster** (back-and-forth lines) across the area.
- An integral term on the *measured* tool tip removes sideways offset.
- The tool axis is kept equal to the *measured* normal.

**Figure (`force_pipe_trace.png`, top panel):** normal force over time on the pipe.
- The dashed black line is the 5 N target; the red line is the measured force.
- The grey vertical lines mark phases:
  - **hover / approach:** 0 N, no contact yet.
  - **sweep:** a short spike to ~5.9 N at first touch, then it stays near 5 N. The small bumps come where the curvature changes along the ±30° arc.
  - **dwell:** hold at the end of the sweep.
  - **lift:** drops to 0.
  - **stow / hold:** arm put away.
- Caption numbers: mean 5.00 N, 0.10 N RMS error, contact never lost, tool only 0.34° from the true normal.

**Table (vessel trace, which feed-forward to use):**
- Columns: mean force, RMS error (typical deviation), and peak (worst moment).
- **desired $F^\ast$** (the one used), **measured $F$** and **none** give the same 5.01 N mean.
- "measured" is slightly worse (0.34 N RMS, 6.6 N peak), because feeding back a noisy measurement adds noise.
- Conclusion: the admittance loop does the work, and the feed-forward doesn't matter. That's an honest negative result.

---

## 8. From depth points to a surface model to a reach plan

**Pipeline (top):**
1. **16 depth frames:** several images averaged, which beats down depth noise.
2. **crop to the work-order frame:** keep only points in the $W\times H$ rectangle next to the marker.
3. **front-most point per cell:** grid the rectangle and keep the closest point in each cell. This removes points seen *behind* the surface.
4. **global thin-plate spline:** fit one smooth surface through all the points.
   - A thin-plate spline is the shape a thin metal sheet takes when bent through the points, i.e. the smoothest surface that fits.
   - "Global" means one fit for the whole area, not pieces stitched together.
5. **bicubic $w(u,v)$ + normals:** resample it onto a fine grid with smooth (bicubic) interpolation, so heights *and* normals (from slopes) are smooth everywhere.
6. **IK over surface: standoff, patches:** inverse kinematics.
   - For each candidate standing distance (standoff), check which surface points the arm can actually reach with the tool normal to the surface.
   - Split the area into patches if one standoff isn't enough.

**Left figure (`surface_model.png`, freeform panel, top-down view in x–y metres):**
- **true surface height:** colour = height in mm (−30 to +25 mm bumps). The red/black wavy line is the planned tool path; the red dots are where the tip actually went.
- **model − true height:** the error of the fitted model. Almost everything is grey, i.e. ≈ 0 on a ±3 mm scale. The caption says 0.03–0.05 mm along the path.
- **model normal error:** the angle between model and true normal, in degrees. It's pale (≈ 0) in the middle and only worse at the very edges, where there are few points.

**Right figure (`reach.pdf`):**
- x-axis is the **standoff**: distance from the robot's base centre to the surface.
- y-axis is the **reachable width**: how many cm of the work strip the arm can cover from there.
- Each line is a surface.
  - Every curve is 0 when too close (the arm folds into itself) and 0 when too far (out of reach).
  - In between there's a plateau: the good standing window.
- **Dashed red (pipe at 0.55 m axis height, the earlier setup):** only one narrow spike around 0.79 m. Any docking error there costs coverage.
- **Solid blue (pipe at 0.66 m):** a wide plateau from 0.64 to 0.70 m, which tolerates dock errors.
- That's why the pipe was raised, and why "plan reach" picks the standoff from this curve rather than a fixed number.

---

## 9. Where am I, and where to look next?

**Localise:**
$$\min_{\Delta x,\Delta y,\Delta\theta}\ \sum_k \rho\big(\mathrm{SD}(R_{\Delta\theta}\,p_k + t)\big)$$
- **$\Delta x,\Delta y,\Delta\theta$:** the correction to the odometry pose (shift in x, shift in y, rotation). These are the unknowns.
- **$p_k$:** the $k$-th point of the depth scan, flattened to the floor plane, in the odometry-guessed position.
- **$R_{\Delta\theta}$:** the 2-D rotation by $\Delta\theta$. **$t = (\Delta x,\Delta y)$:** the shift.
  - $R_{\Delta\theta}p_k + t$ is the scan point after applying the correction.
- **$\mathrm{SD}(\cdot)$:** the signed distance from that point to the nearest map edge (wall or obstacle).
  - If the pose is right, scan points sit exactly on walls, so SD = 0.
  - "Signed" means the value is negative inside obstacles and positive in free space, so the optimiser knows which way to push.
- **$\sum_k$:** add up over all scan points. **$\min$:** find the correction that puts the scan on the walls.
- **$\rho$ (Huber):** a robust cost: quadratic for small errors, linear for big ones.
  - A point that hits something not on the map (a person, a box) gives a big error. Huber stops that one point from dragging the whole solution.
- **Gauss–Newton:** the iterative solver. Linearise, solve, repeat.
- **eigenvalues of $J^\top J$:** $J$ is how the residuals change with $(\Delta x,\Delta y,\Delta\theta)$.
  - In a long featureless corridor the scan pins down sideways position and heading, but not position along the corridor. That direction shows up as a near-zero eigenvalue.
  - The robot only corrects directions with large eigenvalues and keeps odometry for the rest, so it never "corrects" into nonsense.
- **partial-height objects ignored:** the pipe, tank and panel are lower than the scan line or don't fill it, so their 2-D footprint would mislead the matching. They're left out of the map used for localisation.

**Explore:**
$$v^\ast = \arg\max_v\ \text{unseen}(v)\cdot e^{-d(v)/6\,\text{m}}$$
- **$v$:** a candidate viewpoint (a place to stand and look). **$v^\ast$:** the chosen one.
- **$\arg\max_v$:** "the $v$ that gives the largest value" (we want the viewpoint, not the value).
- **$\text{unseen}(v)$:** how much not-yet-seen surface would be visible from $v$, i.e. the information gain.
- **$d(v)$:** walking distance to $v$ (path length on the grid, not straight-line).
- **$e^{-d(v)/6\,\text{m}}$:** a distance discount.
  - At 6 m away a viewpoint is worth $e^{-1} \approx 37\%$ of the same view next to you; at 12 m, about 14 %.
  - 6 m is the length scale that trades "lots to see" against "far to walk".
- **seen = within 25° of the normal and 3.5 m:** a surface patch counts as seen only if viewed nearly head-on (≤ 25° off its normal) and from close enough (≤ 3.5 m). That's what is needed to read a marker.
- **A\* on the grid, stairs = portal edge:** paths are planned with A\* on an occupancy grid. The stairs are a special connecting edge between the two floor areas.

**Top figure (`sdf_map.pdf`):** the localisation map as a signed-distance field, top-down, 25 m × 4.7 m corridor.
- **Dark blue** is far from any wall (≥ 1 m); **white** is right at an edge (0); **red/orange** is inside an obstacle (negative).
- Black outlines are the full-height obstacles used for matching.
- **Orange shapes** are partial-height objects (tank, pipe, panel), which are ignored.
- The dashed box in the middle is the **stairs**.

**Bottom figure (`explore_seed3.png`):** seed 3, what the robot actually did.
- **Blue line:** the path walked, starting at the green dot on the left.
- **Stars:** exploration viewpoints ($v^\ast$ choices). **Purple circles:** look-around spots (stop and turn the head).
- **Green edges:** surfaces seen; **red edges:** not seen.
- **Black arrows "5 (#1, worked)" etc.:** each marker, and the order it was found and worked: 5 → 23 → 7 → 37.
- The brown band is the stairs.

---

## 10. Experimental setup and metrics

**Simulation:**
- **MuJoCo, 2 ms physics:** the physics engine, stepping every 2 ms.
- **WBC 500 Hz, policy 50 Hz:** the WBC runs every physics step; the walking policy every 10th step.
- **torque motors, real limits:** the motors are torque-limited at the real values.
  - Go2 23.7 / 45.4 Nm (hip-thigh / knee).
  - Z1 30 / 60 Nm.
- **25 × 4.7 m corridor, full-width stairs, 4 surfaces:** the world from slide 9.

**Sensor models (realistic noise added on purpose):**
- **RGB 1280×720:** the colour camera, used to read markers. **depth 160×120:** a low-resolution depth camera.
- **depth noise $0.5\,\text{mm} + 2\,\text{mm}\cdot z^2$:** noise grows with the square of distance $z$ (in metres).
  - At 1 m that's 2.5 mm; at 2 m, 8.5 mm.
  - This is how real stereo/ToF cameras behave, and it's why the robot surveys up close.
- **odometry 3 % scale:** walking 1 m reads anywhere from 0.97 to 1.03 m.
- **1 % yaw rate:** turning speed is mis-measured by 1 %.
- **0.11°/m:** heading drifts 0.11° for each metre walked.
- **3 noise seeds × 2 strategies:** each experiment repeated with 3 random noise draws, for each of 2 strategies (work-as-found vs map-first). That's 6 full missions.

**Metrics table:** all measured against simulator ground truth, which the robot never sees.
- **coverage:** share of the requested area actually pressed.
- **force:** mean force, RMS error, and share of time within $5\,\text{N}\pm20\%$ (4–6 N).
- **contact lost:** share of tracing time with $F < 0.5$ N (tool not touching).
- **dock error:** where the robot actually stood vs where it meant to stand.
- **localisation:** estimated vs true position and heading.
- **time:** robot time and distance walked.

---

## 11. Results I: is the controller needed, and what does it need?

**Left figure (`wbc_baselines.pdf`, log scale):** the same precision task under four ways of standing still.
- Spot prototype with arm and a 1.2 kg tool; lower is better in all panels.
- **WBC** (blue): the controller from slide 6.
- **stiff PD:** legs frozen with stiff joint springs and no whole-body reasoning.
- **soft PD:** the same with softer springs. It sags under the arm (227 mm, the seam goes out of reach).
- **RL standing:** the walking policy commanded to stand still. It tilts 28°.
- **Panels:**
  - **body drift [mm]:** WBC 0.29 vs 6.8–97.
  - **body tilt [deg]:** WBC 0.0035° vs 0.74–28°.
  - **tool error RMS [mm]:** WBC 0.015 vs 13–464.
- **Why log scale:** the differences span 3–4 orders of magnitude. On a linear axis the WBC bar would be invisible.
- **"24–850×":** 6.81 / 0.287 ≈ 24× on drift, up to 12.9 / 0.0152 ≈ 850× on tool error, comparing WBC against the *best* baseline (stiff PD).

**Right figure (`ablation.pdf`):** remove one design choice at a time from the final controller and see what breaks. Go2, log x-axes, shorter bar = better, blue = final.
- **Left panel, precision task, tool error:**
  - **QP at 25 Hz:** fell (red bar).
  - **50 Hz:** ~41 mm. **100 Hz:** ~5 mm.
  - So running the controller fast is the single most important thing.
  - default gains, no friction compensation, friction FF without observer, no arm rotor inertia: all in the ~0.7–1.5 mm range. These matter at millimetre level.
  - **unmodelled 1.2 kg tool:** only 0.15 mm. The controller is robust to an unknown tool mass.
- **Right panel, fast arm sweeps, max body drift:**
  - The base-stabilisation choices matter here.
  - A soft base (kp 25 / 100), low damping (kd 1 / 6) or an unmodelled 3 kg tool make the body drift 3–20 mm.
  - Base weight 1 instead of 100, or no foot velocity feedback, barely change anything.
- **Takeaway:** control rate matters more than tuning weights.

---

## 12. Results II: the full mission

**Table:** columns are runs.
- **s1–s3:** "work as found" strategy (do a job as soon as its marker is found), noise seeds 1–3.
- **map first:** explore everything first, then do the jobs in an optimised order. Seeds 1–3, shown as a range.
- Rows:
  - **complete:** all 4 jobs done (✓, and 3/3 for map-first).
  - **time [s]:** total robot time, ~700–775 s.
  - **coverage [%]:** range over the 4 surfaces, 91–100 %.
  - **in 5 N ± 20 % [%]:** share of contact time between 4 and 6 N, 87–100 %.
  - **contact lost:** 0 in every run.
  - **loc. error [cm]:** mean localisation error, 0.8–1.8 cm.

**Bottom-left figure (`paint_seed3.png`):** one panel per marker, seed 3.
- Axes are the work coordinates $u,v$ in metres.
- Colour is the contact force at each spot (colour bar 0–10 N). An even teal colour is ≈ 5 N.
- The **red dashed rectangle** is the requested work-order area.
- The coloured region filling the rectangle means it was covered.
- Each panel title gives coverage and % within 5 N ± 20 %.

**Right figure (`timeline.pdf`):** "where the 705 s go", seed 3.
- One horizontal bar over mission time, coloured by phase:
  - explore 129 s
  - look around 135 s
  - approach + survey 51 s
  - plan reach 0 s (pure computation, no robot time)
  - dock 84 s
  - work 289 s
- "marker N done" labels mark when each job finished.
- Reading: 41 % of the time is spent actually tracing (useful work), and 37 % is searching (explore + look around). Searching is the main cost to cut.

**Photos:** final state at the pipe and the tank. **Video link:** the full seed-3 mission.

**Takeaway:** 4/4 surfaces in all 6 runs, 91–100 % covered at 5 N, contact never lost.

---

## 14. Lessons learned and limitations

**Lessons (each one is a measured surprise):**
- **Control rate before tuning:** from the ablation, 500 Hz works and 25 Hz falls. No amount of gain tuning fixes a slow loop.
- **Test what you believe:** the force feed-forward everyone assumes is essential made no measurable difference (slide 7 table).
- **Curved stickers read at 30° off their normal, not 45°:** a marker on a curved surface is only reliably detected up to 30° viewing angle. That's why "seen" uses 25° on slide 9.
- **2-D footprints mislead scan matching near partial-height objects:** that's why they're ignored in the localisation map.
- **Learned policies stop ~5 cm short:** the RL walker undershoots targets, so docking is closed-loop (measure, correct, repeat).

**Limitations:**
- Simulation only.
- The WBC reads the true state (no estimator).
- The force sensor is ideal (no noise).
- The tool is a frictionless 8 mm ball: contact force is modelled, material removal is not.
- No reaction to obstacles that aren't on the map.
- Pipe docking is 5–10 cm off. The likely cause is marker-pose bias on a curved sticker; it's absorbed by the wide reach window.
- The walking policy wasn't trained with the arm on stairs.

**Sim-to-real needs:** a state estimator, a real force/torque sensor with filtering, a real tool with a process model, and a local obstacle layer.

---

## 15. Conclusion and contributions

1. **Layered autonomy pipeline:** localise → decide → six-step execution, the same for every surface (slide 4).
2. **Stance WBC + arm admittance:** with an ablation showing what the controller actually needs (slides 6, 7, 11).
3. **Map-matching localisation + surface-model reach planning:** evaluated over 3 seeds × 2 strategies (slides 8, 9, 12).

**Built on (credit):** pretrained Go2 walking policies (MoE-CTS), MuJoCo Menagerie robot models, and a teammate's ArUco exploration mission.

**Next:** local obstacle avoidance, a real grinding/inspection tool, walking fine-tuned with the arm on stairs, and hardware.

---

## Backup: full ablation table (Go2)

The numbers behind the left ablation plot. Rows are variants; the first row is the final tuned controller.
- **tool RMS / tool max [mm]:** typical and worst tool position error.
- **body drift [mm]:** how far the body moved while it should have stayed still.
- **foot slip [mm]:** how far the feet slid. It jumps to 213–295 mm at 100/50 Hz: the feet are skating.
- **peak torque:** highest motor torque as a % of its limit. 100 % means a motor was saturated, i.e. out of control authority.

How to read it:
- **"no arm rotor inertia"** saturates torque (100 %) and slips the feet (3.2 mm), even though the tool error is only 0.87 mm. Modelling the motor rotor inertia matters for the legs, not just the arm.
- **The unmodelled 1.2 kg tool** raises peak torque to 77 % but keeps the tool at 0.15 mm.
