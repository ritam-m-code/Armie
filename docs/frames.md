# Frames

Transforms are named `T_<to>_<from>`: `p_base = T_base_cam @ p_cam`.

| Frame | Origin / meaning |
|---|---|
| `base` | Robot base, at the shoulder-pan axis on the mounting surface |
| `cam` | Left stereo camera optical frame (OpenCV: x right, y down, z forward) |
| `ee` | Gripper tip |

All quantities are SI (m, rad, s).
