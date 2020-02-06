"""Class for interfacing with the Kinect2 sensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import numpy as np

try:
    import pylibfreenect2 as lf2
except ImportError:
    logging.warning('Unable to import pylibfreenect2. Python-only Kinect '
                    'driver may not work properly.')

from robovat.perception import image_utils
from robovat.perception import depth_utils
from robovat.perception.camera import Camera


# Constants for Kinect2 camera sensor.
RGB_HEIGHT = 1080
RGB_WIDTH = 1920
DEPTH_HEIGHT = 424
DEPTH_WIDTH = 512
INPAINT_RESCALE_FACTOR = 0.5


class Kinect2PacketPipelineMode:
    """Type of pipeline for Kinect packet processing.
    """
    OPENGL = 0
    CPU = 1


class Kinect2FrameMode:
    """Type of frames that Kinect processes.
    """
    COLOR_DEPTH = 0
    COLOR_DEPTH_IR = 1


class Kinect2RegirobovationMode:
    """Kinect regirobovation mode.
    """
    NONE = 0
    COLOR_TO_DEPTH = 1


class Kinect2DepthMode:
    """Kinect depth mode setting.
    """
    METERS = 0
    MILLIMETERS = 1


class Kinect2(Camera):
    """Class for interacting with a Kinect v2 RGBD sensor.
    """

    def __init__(self,
                 height=DEPTH_HEIGHT,
                 width=DEPTH_WIDTH,
                 intrinsics=None,
                 translation=None,
                 rotation=None,
                 crop=None,
                 packet_pipeline_mode=Kinect2PacketPipelineMode.CPU,
                 regirobovation_mode=Kinect2RegirobovationMode.COLOR_TO_DEPTH,
                 depth_mode=Kinect2DepthMode.METERS,
                 device_num=0,
                 skip_regirobovation=False,
                 use_inpaint=True,
                 upside_down=True):
        """Initialize a Kinect v2 sensor with the given configuration.

        Args:
            packet_pipeline_mode: This indicates packet processing type. Either
                Kinect2PacketPipelineMode.OPENGL or
                Kinect2PacketPipelineMode.CPU.
            regirobovation_mode: The mode for registering a color image to the IR
                camera frame of reference. Either Kinect2RegirobovationMode.NONE
                or Kinect2RegirobovationMode.COLOR_TO_DEPT.
            depth_mode: The units for depths returned from the Kinect frame
                arrays. Either Kinect2DepthMode.METERS or
                Kinect2DepthMode.MILLIMETERS.
            device_num: The sensor's device number on the USB bus.
            skip_regirobovation: If True, the regirobovation step is skipped.
            use_inpaint: If True, inpaint the RGB image and the depth image.
        """
        self._device = None
        self._running = False
        self._packet_pipeline_mode = packet_pipeline_mode
        self._regirobovation_mode = regirobovation_mode
        self._depth_mode = depth_mode
        self._device_num = device_num
        self._skip_regirobovation = skip_regirobovation
        self._use_inpaint = use_inpaint

        # TODO: Why do we need to rotate real-world Kinect2 images in pybullet?
        self._upside_down = upside_down

        super(Kinect2, self).__init__(
            height=height,
            width=width,
            intrinsics=intrinsics,
            translation=translation,
            rotation=rotation,
            crop=crop)

    def __del__(self):
        """Automatically stop the sensor for safety.
        """
        if self.is_running:
            self.stop()

    @property
    def is_running(self):
        """True if the stream is running, or false otherwise.
        """
        return self._running

    def start(self):
        """Starts the Kinect v2 wsensor stream.

        Raises:
            IOError: If the Kinect v2 is not detected.
        """
        # Open packet pipeline.
        if self._packet_pipeline_mode == Kinect2PacketPipelineMode.OPENGL:
            self._pipeline = lf2.OpenGLPacketPipeline()
        elif self._packet_pipeline_mode == Kinect2PacketPipelineMode.CPU:
            self._pipeline = lf2.CpuPacketPipeline()

        # Setup logger.
        self._logger = lf2.createConsoleLogger(lf2.LoggerLevel.Warning)
        lf2.setGlobalLogger(self._logger)

        # check devices
        self._fn_handle = lf2.Freenect2()
        self._num_devices = self._fn_handle.enumerateDevices()

        if self._num_devices == 0:
            raise IOError('Failed to start stream. '
                          'No Kinect2 devices available!')
        if self._num_devices <= self._device_num:
            raise IOError('Failed to start stream. Device num %d unavailable!'
                          % self._device_num)

        # Open device.
        self._serial = self._fn_handle.getDeviceSerialNumber(self._device_num)
        self._device = self._fn_handle.openDevice(self._serial,
                                                  pipeline=self._pipeline)

        # Add device sync modes.
        self._listener = lf2.SyncMultiFrameListener(
            lf2.FrameType.Color | lf2.FrameType.Ir | lf2.FrameType.Depth)
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)

        # Start device.
        self._device.start()

        # Open regirobovation.
        self._regirobovation = None
        if self._regirobovation_mode == Kinect2RegirobovationMode.COLOR_TO_DEPTH:
            logging.debug('Using color to depth regirobovation')
            self._regirobovation = lf2.Regirobovation(
                self._device.getIrCameraParams(),
                self._device.getColorCameraParams())

        self._running = True

    def stop(self):
        """Stops the Kinect2 sensor stream.

        Returns:
            True if the stream was stopped, False if the device was already
                stopped or was not otherwise available.
        """
        # Check that everything is running.
        if not self._running or self._device is None:
            logging.warning('Kinect2 device %d not runnning. Aborting stop.'
                            % self._device_num)
            return False

        # Stop the device.
        self._device.stop()
        self._device.close()
        self._device = None
        self._running = False

        return True

    def _frames(self):
        """Retrieve a new frame from.

        Returns:
            A dictionary of RGB image, depth image and segmentation image.
            'image': The RGB image as an uint8 np array of [width, height, 3].
            'depth': The depth image as a float32 np array of [width, height].

        Raises:
            RuntimeError: If the Kinect stream is not running.
        """
        if not self._running:
            raise RuntimeError('Kinect2 device %s not runnning. '
                               'Cannot read frames' % self._device_num)

        # Read frames.
        frames = self._listener.waitForNewFrame()
        unregistered_color = frames['color']
        distorted_depth = frames['depth']

        # Apply color to depth regirobovation.
        color = unregistered_color
        depth = distorted_depth
        color_depth_map = np.zeros(
                [depth.height, depth.width]
                ).astype(np.int32).ravel()

        if not self._skip_regirobovation and (
                self._regirobovation_mode ==
                Kinect2RegirobovationMode.COLOR_TO_DEPTH):
            depth = lf2.Frame(
                depth.width, depth.height, 4, lf2.FrameType.Depth)
            color = lf2.Frame(
                depth.width, depth.height, 4, lf2.FrameType.Color)
            self._regirobovation.apply(
                unregistered_color, distorted_depth, depth, color,
                color_depth_map=color_depth_map)

        # Convert to array (copy needed to prevent reference of deleted data).
        rgba = copy.copy(color.asarray())

        # Convert BGR to RGB.
        rgba[:, :, [0, 2]] = rgba[:, :, [2, 0]]
        rgba[:, :, 0] = np.fliplr(rgba[:, :, 0])
        rgba[:, :, 1] = np.fliplr(rgba[:, :, 1])
        rgba[:, :, 2] = np.fliplr(rgba[:, :, 2])
        rgba[:, :, 3] = np.fliplr(rgba[:, :, 3])
        rgb = rgba[:, :, :3]

        # Depth image.
        depth = np.fliplr(copy.copy(depth.asarray()))

        # Convert meters.
        if self._depth_mode == Kinect2DepthMode.METERS:
            depth = depth / 1000.0

        # Release and return.
        self._listener.release(frames)

        if self._use_inpaint:
            rgb = image_utils.inpaint(
                rgb, rescale_factor=INPAINT_RESCALE_FACTOR)
            depth = depth_utils.inpaint(
                depth, rescale_factor=INPAINT_RESCALE_FACTOR)

        if self._upside_down:
            rgb = rgb[::-1, ::-1, :]
            depth = depth[::-1, ::-1]

        return {
            'rgb': rgb,
            'depth': depth,
            'segmask': None,
        }
