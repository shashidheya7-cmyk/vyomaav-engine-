"""ROS2 Gazebo Simulation Engine Bridge."""
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from somg.scene import SceneState

@dataclass
class InertiaTensor:
    ixx: float = 1.0
    iyy: float = 1.0
    izz: float = 1.0

    @staticmethod
    def compute_box_inertia(mass: float, dx: float, dy: float, dz: float) -> "InertiaTensor":
        return InertiaTensor(
            ixx=(1/12.0) * mass * (dy**2 + dz**2),
            iyy=(1/12.0) * mass * (dx**2 + dz**2),
            izz=(1/12.0) * mass * (dx**2 + dy**2)
        )

@dataclass
class URDFLink:
    name: str
    mass_kg: float = 1.0
    inertia: InertiaTensor = field(default_factory=InertiaTensor)
    xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    rpy: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    size_box: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    color_rgba: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5, 1.0])
    mesh_ref: Optional[str] = None

@dataclass
class URDFJoint:
    name: str
    parent: str = "world"
    child: str = "base"
    joint_type: str = "revolute"
    origin_xyz: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    parent_link: Optional[str] = None
    child_link: Optional[str] = None
    origin_rpy: Optional[List[float]] = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        if self.parent_link is None:
            self.parent_link = self.parent
        else:
            self.parent = self.parent_link
            
        if self.child_link is None:
            self.child_link = self.child
        else:
            self.child = self.child_link

@dataclass
class URDFRobot:
    name: str
    links: List[URDFLink] = field(default_factory=list)
    joints: List[URDFJoint] = field(default_factory=list)

    def to_urdf_xml(self) -> str:
        xml = f'<?xml version="1.0"?>\n<robot name="{self.name}">\n'
        for link in self.links:
            xml += f'  <link name="{link.name}">\n'
            xml += f'    <visual><geometry><box size="{" ".join(map(str, link.size_box))}"/></geometry></visual>\n'
            xml += f'  </link>\n'
        for j in self.joints:
            xml += f'  <joint name="{j.name}" type="{j.joint_type}">\n'
            xml += f'    <parent link="{j.parent}"/>\n'
            xml += f'    <child link="{j.child}"/>\n'
            xml += f'  </joint>\n'
        xml += '</robot>'
        return xml

class ROS2GazeboBridge:
    @classmethod
    def somg_to_urdf_robot(cls, scene: SceneState, robot_name: Optional[str] = None) -> URDFRobot:
        graph = scene.resolve_active_graph()
        r_name = robot_name or f"{scene.scene_id}_robot"
        robot = URDFRobot(name=r_name)

        for entity_id, entity in graph.nodes.items():
            b_min = entity.spatial.bbox_min
            b_max = entity.spatial.bbox_max
            dx = max(0.01, b_max[0] - b_min[0])
            dy = max(0.01, b_max[1] - b_min[1])
            dz = max(0.01, b_max[2] - b_min[2])
            cx = (b_min[0] + b_max[0]) / 2.0
            cy = (b_min[1] + b_max[1]) / 2.0
            cz = (b_min[2] + b_max[2]) / 2.0
            mass = entity.physics.mass_kg if entity.physics else 1.0
            inertia = InertiaTensor.compute_box_inertia(mass, dx, dy, dz)

            link = URDFLink(
                name=entity_id,
                mass_kg=mass,
                inertia=inertia,
                xyz=[cx, cy, cz],
                size_box=[dx, dy, dz]
            )
            robot.links.append(link)

        for source_id, edges in graph.outgoing_edges.items():
            for edge in edges:
                target_id = edge.target_id
                j_name = f"joint_{source_id}_{target_id}"
                robot.joints.append(URDFJoint(name=j_name, parent=target_id, child=source_id, joint_type="revolute"))

        return robot

    @classmethod
    def export_ros2_package(cls, scene: SceneState, package_name: str, output_dir: str) -> Dict[str, str]:
        pkg_path = os.path.join(output_dir, package_name)
        os.makedirs(pkg_path, exist_ok=True)
        urdf_file = os.path.join(pkg_path, f"{package_name}.urdf")
        world_file = os.path.join(pkg_path, f"{package_name}.world")
        launch_file = os.path.join(pkg_path, f"{package_name}.launch.py")
        pkg_xml_file = os.path.join(pkg_path, "package.xml")
        cmake_file = os.path.join(pkg_path, "CMakeLists.txt")
        
        robot = cls.somg_to_urdf_robot(scene, robot_name=package_name)
        with open(urdf_file, "w") as f:
            f.write(robot.to_urdf_xml())
        with open(world_file, "w") as f:
            f.write('<?xml version="1.0"?><sdf version="1.6"><world name="default"></world></sdf>')
        with open(launch_file, "w") as f:
            f.write(f'# ROS2 Gazebo Launch File for {package_name}\ndef generate_launch_description():\n    # Launch config for {package_name}\n    pass\n')
        with open(pkg_xml_file, "w") as f:
            f.write(f'<?xml version="1.0"?><package format="3"><name>{package_name}</name></package>')
        with open(cmake_file, "w") as f:
            f.write(f'cmake_minimum_required(VERSION 3.8)\nproject({package_name})\n')
            
        return {
            "package_dir": pkg_path,
            "urdf": urdf_file,
            "world": world_file,
            "launch": launch_file,
            "package_xml": pkg_xml_file,
            "cmakelists": cmake_file
        }
