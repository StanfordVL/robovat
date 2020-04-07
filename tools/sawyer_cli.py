#!/usr/bin/env python

"""Command line interface for Sawyer with Kinect2.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import atexit
import os.path
import sys
from builtins import input

import numpy as np
import matplotlib.pyplot as plt
import readline
from mpl_toolkits.mplot3d import Axes3D  # NOQA: For 3D plotting

import _init_paths  # NOQA
from robovat.math import Pose
from robovat.robots import sawyer
from robovat.perception import point_cloud_utils as pc_utils
from robovat.perception.camera import Kinect2
from robovat.simulation import Simulator
from robovat.simulation.camera import BulletCamera
from robovat.utils.yaml_config import YamlConfig
from robovat.utils.logging import logger


WELCOME = (
    '############################################################\n'
    'Command Line Interface\n'
    'Author: Kuan Fang\n'

    'For now, please read the source code for instructions.\n'
    '############################################################\n'
    )

HELP = {
    '############################################################\n'
    'Help'
    '############################################################\n'
    }


CLICK_Z = 0.2


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--mode',
        dest='mode',
        default='sim',
        help='Mode: sim, real.')

    parser.add_argument(
        '--env_config',
        dest='env_config',
        type=str,
        help='The configuration file for the environment.',
        default='configs/envs/arm_env.yaml')

    parser.add_argument(
        '--debug',
        dest='debug',
        type=int,
        help='Use the debugging mode if it is True.',
        default=1)

    parser.add_argument(
        '--assets',
        type=str,
        dest='assets_dir',
        default='./assets',
        help='The assets directory.')

    args = parser.parse_args()

    return args


class EndEffectorClickController(object):
    """Controller of the end effector by mouse clicking."""

    def __init__(self, cli, ax, eef_z=0.2):
        """Initialize.

        Args:
            cli: The command line interface.
            ax: An instance of the Matplotlib Axes.
            eef_z: Z position of the end effector.
        """
        self.cli = cli
        self.image = None
        self.depth = None
        self.eef_z = eef_z

        self.ax = ax
        plt.ion()
        plt.show()

        self.show_image()

    def __call__(self, event):
        """Call.

        Args:
            event: A clicking event.
        """
        pixel = [event.xdata, event.ydata]
        z = self.depth[int(pixel[1]), int(pixel[0])]
        position = self.cli.camera.deproject_pixel(pixel, z)
        pose = Pose([position, [np.pi, 0, 0]])

        position.z = self.eef_z

        while not (self.cli.robot.is_limb_ready() and
                   self.cli.robot.is_gripper_ready()):
            if self.cli.mode == 'sim':
                self.cli.simulator.step()

        print('Clicked pixel: %r. Moving end effector to: % s'
              % (pixel + [z], position))
        self.cli.robot.move_to_gripper_pose(pose)
        while not (self.cli.robot.is_limb_ready() and
                   self.cli.robot.is_gripper_ready()):
            if self.cli.mode == 'sim':
                self.cli.simulator.step()

        self.show_image()
        plt.scatter(pixel[0], pixel[1], c='r')

        return pixel

    def show_image(self):
        """Show the RGB and depth images."""
        self.image = self.cli.camera.frames()['rgb']
        self.depth = self.cli.camera.frames()['depth']
        plt.imshow(self.image)
        plt.title('Image')
        plt.draw()
        plt.pause(1e-3)


class SawyerCLI(object):
    """Command line interface for Sawyer with Kinect2.
    """

    def __init__(self,
                 mode,
                 config,
                 debug,
                 assets_dir):
        """Initialize.

        Args:
            mode: 'sim' or 'real'.
            config: The configuration file for the environment.
            debug: True if it is debugging mode, False otherwise.
            assets_dir: The assets directory.
        """
        print(WELCOME)

        self.mode = mode
        self.config = YamlConfig(config).as_easydict()
        self.debug = debug
        self.assets_dir = assets_dir

        # Command line client input history.
        readline.parse_and_bind('tab: complete')
        history_file = os.path.join('.python_history')

        try:
            readline.read_history_file(history_file)
        except IOError:
            pass

        atexit.register(readline.write_history_file, history_file)

        # Set up the scene.
        if self.mode == 'sim':
            print('Setting up the environment in simulation...')
            self.simulator = Simulator(use_visualizer=self.debug,
                                       assets_dir=self.assets_dir)
            self.simulator.reset()
            self.simulator.start()

            self.ground = self.simulator.add_body(
                self.config.SIM.GROUND.PATH,
                self.config.SIM.GROUND.POSE,
                is_static=True,
                name='ground')

            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table_pose.position.z += np.random.uniform(
                *self.config.SIM.TABLE.HEIGHT_RANGE)
            self.simulator.add_body(
                self.config.SIM.TABLE.PATH,
                self.table_pose,
                is_static=True,
                name='table')

            # Camera.
            self.camera = BulletCamera(
                    simulator=self.simulator,
                    distance=1.0)

        elif self.mode == 'real':
            print('Setting up the environment in the real world...')
            self.table_pose = Pose(self.config.SIM.TABLE.POSE)
            self.table_pose.position.z += np.random.uniform(
                *self.config.SIM.TABLE.HEIGHT_RANGE)
            self.camera = Kinect2(
                    packet_pipeline_mode=0,
                    device_num=0,
                    skip_regirobovation=False,
                    use_inpaint=True)
            self.simulator = None

        else:
            raise ValueError

        # Set up the robot.
        self.robot = sawyer.factory(
                simulator=self.simulator,
                config=self.config.SIM.ARM.CONFIG)

        # Start the camera camera.
        self.camera.set_calibration(
            intrinsics=self.config.KINECT2.DEPTH.INTRINSICS,
            translation=self.config.KINECT2.DEPTH.TRANSLATION,
            rotation=self.config.KINECT2.DEPTH.ROTATION)
        self.camera.start()

        if self.simulator:
            camera_pose = [
                self.config.KINECT2.DEPTH.TRANSLATION,
                self.config.KINECT2.DEPTH.ROTATION]
            camera_pose = Pose(camera_pose).inverse()
            self.simulator.plot_pose(camera_pose, axis_length=0.05)

    def start(self):
        """Start the command line client.
        """
        while (1):
            if self.robot.is_limb_ready() and self.robot.is_gripper_ready():
                sys.stdout.flush()
                command = input('Enter a command: ')

                if command == 'quit' or command == 'q':
                    print('Closing the Sawyer client...')
                    break
                else:
                    self.run_command(command)

            if self.mode == 'sim':
                self.simulator.step()

    def run_command(self, command):  # NOQA
        """Run the input command.

        Args:
            command: An input string command.
        """
        command = command.replace(',', '').replace('[', '').replace(']', '')
        words = command.split(' ')
        command_type = words[0]

        # Print the help information.
        if command_type == 'help' or command_type == 'h':
            print(HELP)

        # Reset the robot joint positions.
        elif command_type == 'reset' or command_type == 'r':
            self.robot.reset(self.config.ARM.OFFSTAGE_POSITIONS)

        # Visualize the camera image.
        elif command_type == 'visualize' or command_type == 'v':
            results = self.camera.frames()
            image = results['rgb']
            depth = results['depth']

            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.imshow(image)
            plt.title('RGB Image')
            plt.subplot(122)
            plt.imshow(depth)
            plt.title('Depth Image')

            plt.show()

        # Visualize the table.
        elif command_type == 'table' or command_type == 't':
            results = self.camera.frames()
            image = results['rgb']
            depth = results['depth']

            table_points = [
                [0, 0, 0],
                [0, -0.61, 0],
                [0, 0.61, 0],
                [-0.38, 0, 0],
                [0.38, 0, 0],
                [-0.38, -0.61, 0],
                [-0.38, 0.61, 0],
                [0.38, -0.61, 0],
                [0.38, 0.61, 0],
            ]
            table_offset = self.table_pose.position
            table_points = np.array(table_points) + table_offset
            table_pixels = self.camera.project_point(table_points)

            plt.figure(figsize=(20, 10))
            plt.subplot(121)
            plt.imshow(image)
            plt.scatter(table_pixels[:, 0], table_pixels[:, 1], c='r')
            plt.title('RGB Image')
            plt.subplot(122)
            plt.imshow(depth)
            plt.scatter(table_pixels[:, 0], table_pixels[:, 1], c='r')
            plt.title('Depth Image')

            plt.show()

        # Visualize the layout.
        elif command_type == 'layout' or command_type == 'l':
            results = self.camera.frames()
            image = results['rgb']

            layout_name = words[1]
            layout_config = self.config.LAYOUT[layout_name]

            tile_config = layout_config.REGION
            size = layout_config.SIZE
            offset = layout_config.OFFSET

            plt.figure(figsize=(10, 10))
            plt.subplot(111)
            plt.imshow(image)

            for i, center in enumerate(tile_config.CENTERS):
                position = np.array(offset) + np.array(center) * size
                x = position[0]
                y = position[1]
                z = self.table_pose.position.z
                corners = [
                    [x - 0.5 * size, y - 0.5 * size, z],
                    [x + 0.5 * size, y - 0.5 * size, z],
                    [x - 0.5 * size, y + 0.5 * size, z],
                    [x + 0.5 * size, y + 0.5 * size, z]]

                pixels = self.camera.project_point(corners)

                color = 'green'
                plt.plot([pixels[0, 0], pixels[1, 0]],
                         [pixels[0, 1], pixels[1, 1]],
                         color=color, linewidth=2)
                plt.plot([pixels[0, 0], pixels[2, 0]],
                         [pixels[0, 1], pixels[2, 1]],
                         color=color, linewidth=2)
                plt.plot([pixels[1, 0], pixels[3, 0]],
                         [pixels[1, 1], pixels[3, 1]],
                         color=color, linewidth=2)
                plt.plot([pixels[2, 0], pixels[3, 0]],
                         [pixels[2, 1], pixels[3, 1]],
                         color=color, linewidth=2)

            plt.show()

        # Visualize the camera image.
        elif command_type == 'pointcloud' or command_type == 'pc':
            if len(words) == 1:
                num_clusters = 0
            else:
                num_clusters = int(words[1])

            images = self.camera.frames()
            image = images['rgb']
            depth = images['depth']
            point_cloud = self.camera.deproject_depth_image(depth)

            fig = plt.figure(figsize=(20, 5))

            ax1 = fig.add_subplot(141)
            ax1.imshow(depth)

            if (self.config.OBS.CROP_MIN is not None and
                    self.config.OBS.CROP_MAX is not None):
                crop_max = np.array(self.config.OBS.CROP_MAX)[np.newaxis, :]
                crop_min = np.array(self.config.OBS.CROP_MIN)[np.newaxis, :]
                crop_mask = np.logical_and(
                    np.all(point_cloud >= crop_min, axis=-1),
                    np.all(point_cloud <= crop_max, axis=-1))
                point_cloud = point_cloud[crop_mask]

            ax2 = fig.add_subplot(142, projection='3d')
            downsampled_point_cloud = pc_utils.downsample(
                    point_cloud, num_samples=4096)
            pc_utils.show(downsampled_point_cloud, ax2, axis_range=1.0)

            if num_clusters > 0:
                point_cloud = pc_utils.remove_table(point_cloud)

                segmask = pc_utils.cluster(
                    point_cloud, num_clusters=num_clusters, method='dbscan')
                point_cloud = point_cloud[segmask != -1]

                segmask = pc_utils.cluster(
                    point_cloud, num_clusters=num_clusters)
                point_cloud = pc_utils.group_by_labels(
                    point_cloud, segmask, num_clusters, 256)

                ax3 = fig.add_subplot(143, projection='3d')
                pc_utils.show(point_cloud, ax3, axis_range=1.0)

                ax4 = fig.add_subplot(144)
                pc_utils.show2d(point_cloud, self.camera, ax4, image=image)

            plt.show()

        # Visualize the camera image.
        elif command_type == 'rgbd':
            images = self.camera.frames()
            image = images['rgb']
            depth = images['depth']
            point_cloud = self.camera.deproject_depth_image(depth)

            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)

            if (self.config.OBS.CROP_MIN is not None and
                    self.config.OBS.CROP_MAX is not None):
                crop_max = np.array(self.config.OBS.CROP_MAX)[np.newaxis, :]
                crop_min = np.array(self.config.OBS.CROP_MIN)[np.newaxis, :]
                crop_mask = np.logical_and(
                    np.all(point_cloud >= crop_min, axis=-1),
                    np.all(point_cloud <= crop_max, axis=-1))
                point_cloud = point_cloud[crop_mask]

            point_cloud = pc_utils.remove_table(point_cloud)
            point_cloud = pc_utils.downsample(point_cloud, num_samples=4096)
            pixels = self.camera.project_point(point_cloud)
            pixels = np.array(pixels, dtype=np.int32)

            background = np.zeros_like(image)
            background[pixels[:, 1], pixels[:, 0]] = (
                image[pixels[:, 1], pixels[:, 0]])

            plt.imshow(background)

            plt.show()

        # Move the gripper to the clicked pixel position.
        elif command_type == 'click' or command_type == 'c':

            fig, ax = plt.subplots(figsize=(20, 10))
            onclick = EndEffectorClickController(self, ax)

            results = self.camera.frames()
            plt.imshow(results['rgb'])
            plt.title('Image')
            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.show()

        # Move joints to the target positions.
        elif command_type == 'joints' or command_type == 'j':
            joint_positions = [float(ch) for ch in words[1:]]
            print('Moving to joint positions: %s ...' % joint_positions)
            self.robot.move_to_joint_positions(joint_positions)

        # Move the end effector to the target pose.
        elif command_type == 'end_effector' or command_type == 'e':
            pose = [float(ch) for ch in words[1:]]
            if len(pose) == 6 or len(pose) == 7:
                pose = Pose(pose[:3], pose[3:])
            elif len(pose) == 3:
                end_effector_pose = self.robot.end_effector
                pose = Pose([pose, end_effector_pose.orientation])
            else:
                print('The format of the input pose is wrong.')

            print('Moving to end effector pose: %s ...' % pose)
            self.robot.move_to_gripper_pose(pose)

        # Move the end effector to the target pose.
        elif command_type == 'end_effector_line' or command_type == 'el':
            pose = [float(ch) for ch in words[1:]]
            if len(pose) == 6 or len(pose) == 7:
                pose = Pose(pose[:3], pose[3:])
            elif len(pose) == 3:
                end_effector_pose = self.robot.end_effector
                pose = Pose(pose, end_effector_pose.orientation)
            else:
                print('The format of the input pose is wrong.')

            print('Moving to end effector pose: %s ...' % pose)
            self.robot.move_to_gripper_pose(pose, straight_line=True)

        # Open the gripper.
        elif command_type == 'open' or command_type == 'o':
            joint_positions = self.robot.grip(0)

        # Close the gripper.
        elif command_type == 'grasp' or command_type == 'g':
            joint_positions = self.robot.grip(1)

        # Print the current robot status.
        elif command_type == 'print' or command_type == 'p':
            joint_positions = self.robot.joint_positions
            joint_positions = [
                    joint_positions['right_j0'],
                    joint_positions['right_j1'],
                    joint_positions['right_j2'],
                    joint_positions['right_j3'],
                    joint_positions['right_j4'],
                    joint_positions['right_j5'],
                    joint_positions['right_j6'],
                    ]
            print('Joint positions: %s' % (joint_positions))

            end_effector_pose = self.robot.end_effector
            print('End Effector position: %s, %s' %
                  (end_effector_pose.position, end_effector_pose.euler))

        else:
            print('Unrecognized command: %s' % command)


def main():
    args = parse_args()

    logger.info('Creating the Sawyer command line client...')
    sawyer_cli = SawyerCLI(
            args.mode,
            args.env_config,
            args.debug,
            args.assets_dir)

    logger.info('Running the Sawyer command line client...')
    sawyer_cli.start()


if __name__ == '__main__':
    main()
