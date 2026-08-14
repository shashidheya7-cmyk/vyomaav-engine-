"""
VYOMAAV Base Model Engine
Test Suite: tests/test_ros2.py

Pytest / Unittest suite validating Sprint 17: URDF link and joint generation, inertia tensor calculations,
ROS2 Gazebo .world SDF export, and complete ROS2 workspace package generation.
"""

import os
import xml.etree.ElementTree as ET

from somg.scene import SceneState
from somg.entity import SOMGEntity, SpatialComponent, PhysicsComponent, MaterialComponent
from somg.graph import RelationType
from engine.ros2 import InertiaTensor, URDFRobot, URDFLink, URDFJoint, ROS2GazeboBridge


def test_inertia_tensor_calculation():
    # Box mass=12.0 kg, dx=2.0, dy=1.0, dz=0.5
    # ixx = 1/12 * 12 * (1^2 + 0.5^2) = 1.25
    # iyy = 1/12 * 12 * (2^2 + 0.5^2) = 4.25
    # izz = 1/12 * 12 * (2^2 + 1^2) = 5.00
    inertia = InertiaTensor.compute_box_inertia(12.0, 2.0, 1.0, 0.5)

    assert abs(inertia.ixx - 1.25) < 1e-4
    assert abs(inertia.iyy - 4.25) < 1e-4
    assert abs(inertia.izz - 5.00) < 1e-4


def test_urdf_robot_xml_serialization():
    robot = URDFRobot(name="test_bot")
    link_a = URDFLink(
        name="base_link",
        mass_kg=5.0,
        inertia=InertiaTensor(ixx=0.5, iyy=0.5, izz=0.5),
        xyz=[0.0, 0.0, 0.0],
        rpy=[0.0, 0.0, 0.0],
        size_box=[1.0, 1.0, 1.0]
    )
    link_b = URDFLink(
        name="arm_link",
        mass_kg=2.0,
        inertia=InertiaTensor(ixx=0.1, iyy=0.1, izz=0.1),
        xyz=[0.0, 0.0, 0.5],
        rpy=[0.0, 0.0, 0.0],
        size_box=[0.2, 0.2, 0.8]
    )
    joint_ab = URDFJoint(
        name="shoulder_joint",
        joint_type="revolute",
        parent_link="base_link",
        child_link="arm_link",
        origin_xyz=[0.0, 0.0, 0.5],
        origin_rpy=[0.0, 0.0, 0.0]
    )

    robot.links.extend([link_a, link_b])
    robot.joints.append(joint_ab)

    xml_str = robot.to_urdf_xml()

    assert '<robot name="test_bot">' in xml_str
    assert '<link name="base_link">' in xml_str
    assert '<joint name="shoulder_joint" type="revolute">' in xml_str

    # Validate XML syntax parseability
    root = ET.fromstring(xml_str)
    assert root.tag == "robot"
    assert len(root.findall("link")) == 2
    assert len(root.findall("joint")) == 1


def test_somg_to_urdf_robot_conversion():
    scene = SceneState(scene_id="RoboticArmScene")

    base = SOMGEntity(
        entity_id="base",
        spatial=SpatialComponent(bbox_min=[-0.5, 0.0, -0.5], bbox_max=[0.5, 0.2, 0.5]),
        physics=PhysicsComponent(mass_kg=20.0, is_static=True)
    )
    arm = SOMGEntity(
        entity_id="arm",
        spatial=SpatialComponent(bbox_min=[-0.1, 0.2, -0.1], bbox_max=[0.1, 1.0, 0.1]),
        physics=PhysicsComponent(mass_kg=3.5, is_static=False)
    )

    scene.base_graph.add_node(base)
    scene.base_graph.add_node(arm)
    scene.base_graph.add_edge("arm", "base", RelationType.ATTACHED_TO)

    robot = ROS2GazeboBridge.somg_to_urdf_robot(scene)

    assert robot.name == "RoboticArmScene_robot"
    assert len(robot.links) == 2
    assert len(robot.joints) == 1
    assert robot.joints[0].joint_type == "revolute"
    assert robot.joints[0].parent_link == "base"
    assert robot.joints[0].child_link == "arm"


def test_ros2_package_export(tmp_path):
    scene = SceneState(scene_id="LabScene")
    robot_entity = SOMGEntity("mobile_robot", spatial=SpatialComponent(bbox_min=[-0.5, 0.0, -0.5], bbox_max=[0.5, 0.8, 0.5]))
    scene.base_graph.add_node(robot_entity)

    out_dir = str(tmp_path)
    files = ROS2GazeboBridge.export_ros2_package(scene, package_name="vyomaav_bot", output_dir=out_dir)

    assert os.path.exists(files["package_dir"])
    assert os.path.exists(files["urdf"])
    assert os.path.exists(files["world"])
    assert os.path.exists(files["launch"])
    assert os.path.exists(files["package_xml"])
    assert os.path.exists(files["cmakelists"])

    # Check content of launch file
    with open(files["launch"], "r") as f:
        launch_code = f.read()
    assert "generate_launch_description" in launch_code
    assert "vyomaav_bot" in launch_code


if __name__ == "__main__":
    test_inertia_tensor_calculation()
    test_urdf_robot_xml_serialization()
    test_somg_to_urdf_robot_conversion()
    test_ros2_package_export("tmp_ros2_test")
    print("ALL TESTS PASSED SUCCESSFULLY!")