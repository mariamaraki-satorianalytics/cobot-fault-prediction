# The UR3 collaborative robot: a primer

This page explains the machine behind the data, so the features and modelling choices in this project read as engineering decisions rather than arbitrary ones. The README has a shorter version. Every claim links to its source (listed at the end).

## 1. Collaborative robots

A **collaborative robot (cobot)** is a robot intended for direct, close-range interaction with people [[1]](#sources). Classic industrial robots are kept away from operators, typically behind fences or light curtains.

What lets a cobot share space with people is **power and force limiting (PFL)**:
- The controller continuously limits the forces and pressures the arm can exert to levels considered safe for human contact, as defined in ISO/TS 15066. The robot's own safety requirements come from ISO 10218-1 [[2]](#sources)[[3]](#sources).
- Limited quantities include joint positions and speeds, tool and elbow speed and force, momentum and power [[3]](#sources).
- Exceeding any of these limits makes the robot **stop** [[2]](#sources).

Whether an application still needs guarding is decided by its own risk assessment, not by the robot alone [[2]](#sources).

## 2. The UR3

The UR3 is Universal Robots' compact, table-top arm. Like the rest of the UR family it is a **six-axis serial arm**: six rotating joints in a chain from the base to the tool. Six joints are what an arm needs to reach a point in space (3 degrees of freedom) *and* hold the tool at any orientation there (3 more).

<p align="center">
  <img src="img/ur5_cobot_growskills.jpg" alt="A Universal Robots UR5 cobot arm with a gripper, mounted on a flight case with its teach pendant below" height="340">
  &nbsp;
  <img src="img/ur3_joints.svg" alt="UR3 joint layout J0 to J5" width="520">
</p>
<p align="center"><sub>
  <b>Left:</b> a UR5 from the same Universal Robots family (the UR3 is the smaller model with the same six-joint layout).
  Photo by GrowSkills Robotics, <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a>, via <a href="https://commons.wikimedia.org/wiki/File:Cobot.jpg">Wikimedia Commons</a> (cropped and resized).
  <b>Right:</b> simplified joint layout, J0–J5 as named in the dataset.
</sub></p>

| Joint (dataset) | UR name | Role |
|---|---|---|
| J0 | Base | Rotates the whole arm about the vertical axis |
| J1 | Shoulder | Lifts the upper arm; carries the most gravity load |
| J2 | Elbow | Lifts the forearm |
| J3 | Wrist 1 | Wrist pitch |
| J4 | Wrist 2 | Wrist rotation about a perpendicular axis |
| J5 | Wrist 3 | Tool flange rotation; carries the gripper |

UR's own interfaces report joint values in the order **[Base, Shoulder, Elbow, Wrist 1, Wrist 2, Wrist 3]** [[4]](#sources). The dataset's J0–J5 follow the same order. The dataset doesn't state this, but the holding-current analysis in the EDA notebook supports it: the shoulder draws the most current at rest, then the elbow.

### Specifications

| | UR3 (CB3 series) [[5]](#sources) | UR3e (e-Series) [[6]](#sources) |
|---|---|---|
| Joints | 6 rotating | 6 rotating |
| Payload | 3 kg | 3 kg |
| Reach | 500 mm | 500 mm |
| Weight | 11 kg | 11.2 kg |
| Pose repeatability | ±0.1 mm | ±0.03 mm (ISO 9283) |
| Joint range | ±360°, infinite on the last joint | ±360° (base to wrist 2), infinite on wrist 3 |
| Joint speed | wrists 360°/s, other joints 180°/s | wrists 360°/s, base/shoulder/elbow 180°/s |

**Which one is in the dataset?** Neither the dataset page nor the paper names the series [[7]](#sources)[[8]](#sources). The paper describes the robot as having a 500 mm reach and ±0.1 mm repeatability, which matches the CB3 UR3 rather than the UR3e. That is an inference, not a confirmed fact.

### Why current is a window into the mechanics

Each joint is driven by an electric motor. For the brushless permanent-magnet motors typical of robot joints, output torque is approximately proportional to winding current, **τ ≈ kₜ · I**. So the joint currents in this dataset are a (noisy, scaled) measurement of the torque each joint is producing. That torque has three sources:

- **Gravity:** holding the arm's own weight and the payload. This depends on the pose, so it explains why the shoulder and elbow draw current even at standstill.
- **Inertia:** accelerating and decelerating the links. This produces the spikes during motion.
- **External forces:** contact, collisions or an unexpected payload. This is the part that matters for faults.

Temperatures follow from the current through resistive heating (P = I²R) and change slowly. Speeds tell us whether a joint is moving at all.

## 3. Protective stops

A **protective stop** is issued when the controller detects that the planned trajectory cannot be followed [[9]](#sources). The arm decelerates along its path and halts. Typical causes listed by UR:

- collisions or unexpected contact;
- a payload or mounting that doesn't match the configured settings;
- cables or hoses dragging on the robot;
- joint-limit, tool-orientation or safety-plane violations;
- very high accelerations with blended moves;
- cold starts.

The controller reports the reason as a code. For example, C150 is a joint limit, C152 a safety plane, C153/C159 path deviation and C157/C158 joint collision [[9]](#sources).

| | Protective stop | Emergency stop |
|---|---|---|
| Trigger | Controller can't follow the path, or a limit is exceeded | A person presses an e-stop device |
| Motor power | Stays on | Removed, brakes engage |
| Recovery | A deliberate operator action (automatic reset is not allowed) | Manual reset and restart |
| Sources | [[9]](#sources) | [[10]](#sources) |

Each protective stop costs cycle time and needs a person to step in. Seeing one coming a few seconds early is the business case for this project.

> The paper's variable table describes `Robot_ProtectiveStop` as an "emergency halt status" [[8]](#sources). Following UR's terminology, this project treats it as a protective stop. The two are different mechanisms.

## 4. How the data was collected

The dataset was logged through two of the controller's standard interfaces [[7]](#sources):

- **RTDE (Real-Time Data Exchange)** synchronises external applications with the UR controller over TCP/IP, on port 30004 [[11]](#sources). It exposes 6-vectors such as `actual_current`, `joint_temperatures` and `actual_qd` (joint speeds) [[12]](#sources).
- **MODBUS TCP:** the controller runs a Modbus server on port 502 with general-purpose registers [[13]](#sources).

RTDE can stream far faster than this dataset's rate; the paper mentions 125 Hz [[8]](#sources). The published file is sampled at about **1 Hz**, which limits how fine-grained any precursor signal can be.

## Summary

- The UR3 is a six-joint arm (J0–J5 = base to wrist 3) that shares space with people because its controller polices forces and speeds.
- When the controller can't follow the planned path, it raises a **protective stop**.
- Joint currents act as torque sensors, temperatures integrate electrical load over time, and speeds show motion.
- Together they describe what the arm is doing and what it is pushing against. A model can therefore look for the signature of trouble in those signals *before* the stop happens.

## Sources

1. Wikipedia, "Cobot". https://en.wikipedia.org/wiki/Cobot
2. Universal Robots, "Safety FAQ". https://www.universal-robots.com/articles/ur/safety/safety-faq/
3. Universal Robots, "Safety paper: the standard we set". https://www.universal-robots.com/articles/ur/safety/safety-paper-the-standard-we-set/
4. Universal Robots, URScript manual, `get_actual_joint_positions()`. https://www.universal-robots.com/manuals/EN/HTML/SW5_24/Content/prod-scriptmanual/all_scripts/get_actual_joint_positio.htm
5. Universal Robots, UR3 technical specifications (CB3, 2015). https://www.universal-robots.com/media/240787/ur3_us.pdf
6. Universal Robots, UR3e datasheet. https://www.universal-robots.com/manuals/latest/en/datasheets/ur3e/
7. UCI Machine Learning Repository, "UR3 CobotOps". https://archive.ics.uci.edu/dataset/963/ur3+cobotops (DOI [10.24432/C5J891](https://doi.org/10.24432/C5J891))
8. Tyrovolas, Stylios, Aliev, Antonelli, "Leveraging Information Flow-Based Fuzzy Cognitive Maps for Interpretable Fault Diagnosis in Industrial Robotics", DoCEIS 2024, IFIP AICT 716, pp. 98–110. https://doi.org/10.1007/978-3-031-63851-0_6
9. Universal Robots, "Understanding protective stops". https://www.universal-robots.com/articles/ur/robot-care-maintenance/understanding-protective-stops/
10. Universal Robots, "Connecting an emergency stop device vs. a safeguard protective device". https://www.universal-robots.com/articles/ur/application-installation/connecting-an-emergency-stop-device-vs-a-safeguard-protective-device/
11. Universal Robots, "Overview of client interfaces". https://www.universal-robots.com/articles/ur/interface-communication/overview-of-client-interfaces/
12. Universal Robots, "RTDE guide". https://docs.universal-robots.com/tutorials/communication-protocol-tutorials/rtde-guide.html
13. Universal Robots, "Modbus server". https://www.universal-robots.com/articles/ur/interface-communication/modbus-server/
