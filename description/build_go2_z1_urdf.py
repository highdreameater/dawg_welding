"""Assemble a combined Unitree Go2 + Z1 URDF from the *official* Unitree descriptions (unitree_ros).

Unitree publishes no Go2+Z1 model. Their only quadruped+Z1 description is aliengoZ1_description, which mounts the
arm with a fixed joint `trunk_arm_joint` (trunk -> link00) at
    x = trunk_length/2 - 0.07,  y = 0,  z = trunk_height/2,  rpy = 0
This script applies the same rule to Go2 (trunk 0.3762 x 0.0935 x 0.114 m -> x = 0.1181, z = 0.057) and copies all
links/joints/inertials verbatim from:
    unitree_ros/robots/go2_description/urdf/go2_description.urdf
    unitree_ros/robots/z1_description/xacro/z1.urdf            (Z1 without gripper; drop the world link)
Mesh URIs package://<pkg>/... are rewritten to paths relative to the output file.

    python go2_z1/description/build_go2_z1_urdf.py            # -> go2_z1/description/go2_z1.urdf
"""
import argparse
import copy
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROBOTS = os.path.join(HERE, "upstream", "unitree_ros", "robots")
GO2 = os.path.join(ROBOTS, "go2_description", "urdf", "go2_description.urdf")
Z1 = os.path.join(ROBOTS, "z1_description", "xacro", "z1.urdf")
TRUNK_LENGTH, TRUNK_HEIGHT = 0.3762, 0.114  # go2_description/xacro/const.xacro


def rewrite_meshes(root, out_dir):
    for m in root.iter("mesh"):
        fn = m.get("filename", "")
        if fn.startswith("package://"):
            pkg, rel = fn[len("package://"):].split("/", 1)
            m.set("filename", os.path.relpath(os.path.join(ROBOTS, pkg, rel), out_dir))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "go2_z1.urdf"))
    args = ap.parse_args()
    out_dir = os.path.dirname(os.path.abspath(args.out))

    go2 = ET.parse(GO2).getroot()
    z1 = ET.parse(Z1).getroot()
    robot = ET.Element("robot", {"name": "go2_z1"})
    robot.append(ET.Comment(" Assembled by build_go2_z1_urdf.py from official unitree_ros go2_description + z1_description; "
                            "mount rule from unitree_ros aliengoZ1_description (trunk_arm_joint). "))

    for el in go2:
        if el.tag in ("link", "joint", "material"):
            robot.append(copy.deepcopy(el))

    ax, az = TRUNK_LENGTH / 2.0 - 0.07, TRUNK_HEIGHT / 2.0
    mount = ET.SubElement(robot, "joint", {"name": "trunk_arm_joint", "type": "fixed"})
    ET.SubElement(mount, "origin", {"rpy": "0 0 0", "xyz": f"{ax:.4f} 0 {az:.4f}"})
    ET.SubElement(mount, "parent", {"link": "base"})
    ET.SubElement(mount, "child", {"link": "link00"})

    for el in z1:
        if el.tag == "link" and el.get("name") == "world":
            continue
        if el.tag == "joint" and el.get("name") == "base_static_joint":
            continue
        if el.tag in ("link", "joint", "material"):
            robot.append(copy.deepcopy(el))

    rewrite_meshes(robot, out_dir)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(args.out, encoding="utf-8", xml_declaration=True)
    print(f"wrote {args.out}  (arm mount x={ax:.4f} z={az:.4f} on 'base')")


if __name__ == "__main__":
    main()
