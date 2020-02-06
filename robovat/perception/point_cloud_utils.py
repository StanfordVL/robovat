"""Utilities for point cloud data."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from sklearn import cluster as sklearn_cluster

from robovat.utils.logging import logger

try:
    import pcl
except ImportError:
    logger.warning('Failed to import pcl')


COLOR_MAP = ['r', 'y', 'b', 'g', 'm', 'k']


def downsample(point_cloud, num_samples):
    """Downsample a point cloud.

    Args:
        point_cloud: 3D point cloud of shape [num_points, 3].
        num_samples: Number of points to keep.

    Returns:
        downsampled_point_cloud: Downsampled 3D point cloud of shape
            [num_samples, 3].
    """
    num_points = point_cloud.shape[0]
    replace = num_points < num_samples
    inds = np.random.choice(np.arange(num_points), size=num_samples,
                            replace=replace)
    downsampled_point_cloud = point_cloud[inds]
    return downsampled_point_cloud


def segment_by_ids(point_cloud, segmask, body_ids, num_samples):
    """Downsample a point cloud.

    Args:
        point_cloud: 3D point cloud of shape [num_points, 3].
        segmask: Integer array of shape [num_points].
        body_ids: List of segmentation IDs of the target bodies.
        num_samples: Number of points to keep for each body.

    Returns:
        segmented_point_cloud: Segmented 3D point cloud of shapeverageaverage
            [num_bodies, num_samples, 3].
    """
    num_bodies = len(body_ids)
    segmented_point_cloud = np.zeros([num_bodies, num_samples, 3],
                                     dtype=np.float32)

    for i in range(num_bodies):
        body_id = body_ids[i]
        inds_i = np.where(segmask == body_id)[0]
        if len(inds_i) > 0:
            point_cloud_i = point_cloud[inds_i]
            point_cloud_i = downsample(point_cloud_i, num_samples)
            segmented_point_cloud[i] = point_cloud_i
        else:
            logger.warning('No points were found for object % d.' % i)

    return segmented_point_cloud


def cluster(point_cloud,
            num_clusters,
            method='agg'):
    """Cluster point cloud.

    Args:
        point_cloud: 3D point cloud of shape [num_points, 3].
        num_clusters: Number of clusters.
        method: Clustering algorithms.

    Returns:
        segmask: The segmentation mask as an integer array.
    """
    if method == 'dbscan':
        algorithm = sklearn_cluster.DBSCAN(
            min_samples=50,
            eps=0.02)

    elif method == 'agg':
        connectivity = kneighbors_graph(
            point_cloud, n_neighbors=100, include_self=False)
        connectivity = 0.5 * (connectivity + connectivity.T)
        algorithm = sklearn_cluster.AgglomerativeClustering(
            linkage='ward',
            affinity='euclidean',
            n_clusters=num_clusters,
            connectivity=connectivity)

    else:
        raise ValueError

    algorithm.fit(point_cloud)
    segmask = algorithm.labels_.astype(np.int)

    # TODO: Check if found enough clusters.
    return segmask


def convert_segment_ids(segmask, body_ids):
    """Convert IDs in the segmentation mask.

    Replace the IDs in the body IDs in the segmentation masks into the
    corresponding indices in the given list of body IDs.

    Args:
        segmask: The segmentation mask as an integer array.
        body_ids: A list of IDs.

    Returns:
        The converted segmentation mask.
    """
    new_segmask = -1 * np.ones_like(segmask)

    num_bodies = len(body_ids)
    for i in range(num_bodies):
        body_id = body_ids[i]
        inds_i = np.where(segmask == body_id)[0]
        new_segmask[inds_i] = i

    return new_segmask


def group_by_labels(point_cloud, segmask, num_clusters, num_samples):
    """Group a point cloud by the segmentation mask.

    Args:
        point_cloud: 3D point cloud of shape [num_points, 3].
        segmask: Integer array of shape [num_points].
        num_clusters: Number of clusters.
        num_samples: Number of points to keep for each body.

    Returns:
        segmented_point_cloud: Segmented 3D point cloud of shape
            [num_clusters, num_samples, 3].
    """
    segmented_point_cloud = np.zeros([num_clusters, num_samples, 3],
                                     dtype=np.float32)

    for i in range(num_clusters):
        inds_i = np.where(segmask == i)[0]
        if len(inds_i) > 0:
            point_cloud_i = point_cloud[inds_i]
            point_cloud_i = downsample(point_cloud_i, num_samples)
            segmented_point_cloud[i] = point_cloud_i

    return segmented_point_cloud


def remove_table(point_cloud, thresh=0.015):
    """Remove all table points using ransac.

    Args:
        point_cloud: (num_points x 3) 3D point cloud.
        thresh: Maximum distance to allow in ransac.

    Returns:
        segmented_cloud: 3D point cloud without table points of shape
            [num_segmented_points, 3].
    """
    # PCL version.
    num_points = point_cloud.shape[0]
    cloud = pcl.PointCloud(point_cloud.astype(np.float32))
    segmenter = cloud.make_segmenter()
    segmenter.set_model_type(pcl.SACMODEL_PLANE)
    segmenter.set_method_type(pcl.SAC_RANSAC)
    segmenter.set_distance_threshold(thresh)
    indices, model = segmenter.segment()
    obj_idxs = np.setdiff1d(np.arange(num_points), indices)
    segmented_cloud = point_cloud[obj_idxs]

    # PyntCloud version.
    # cloud = PyntCloud(pd.DataFrame({'x': point_cloud[:, 0],
    #                                 'y': point_cloud[:, 1],
    #                                 'z': point_cloud[:, 2]}))
    # cloud.add_scalar_field("plane_fit", max_dist=thresh)
    # point_cloud = cloud.points.values
    # segmented_cloud = point_cloud[point_cloud[:, -1] == 0][:, :3]

    return segmented_cloud


def show(point_cloud,
         ax=None,
         c=None,
         axis_limit=None,
         axis_range=None):
    """Plot point cloud and return subplot handle.

    Usage of plotting point cloud in an 3D coordinate system:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.title('Point cloud')
        ax = fig.add_subplot(111, projection='3d')
        plot(ax, point_cloud)
        plt.show()

    Args:
        point_cloud: The point cloud as an array.
        ax: The matplotlib ax.
        c: Color of the point cloud.
        axis_limit: Limit of the axes.
        axis_range: Range of the axes.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.cla()

    # Plot.
    if len(point_cloud.shape) == 2:
        # Plot a holistic point cloud.
        if c is None:
            c = 'b'
        xs = point_cloud[:, 0]
        ys = point_cloud[:, 1]
        zs = point_cloud[:, 2]
        ax.scatter(xs, ys, zs, s=1, c=c)
    elif len(point_cloud.shape) == 3:
        # Plot segmented point cloud.
        if c is None:
            c = COLOR_MAP
        for i in range(point_cloud.shape[0]):
            xs = point_cloud[i, :, 0]
            ys = point_cloud[i, :, 1]
            zs = point_cloud[i, :, 2]
            color = c[i]
            ax.scatter(xs, ys, zs, s=1, c=color)
    else:
        raise ValueError

    # Set axes.
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    if axis_limit is None and axis_range is None:
        ax.set_aspect('equal')
    elif axis_range is not None:
        if len(point_cloud.shape) == 2:
            avg_pos = np.mean(point_cloud, axis=0)
        elif len(point_cloud.shape) == 3:
            avg_pos = np.mean(np.mean(point_cloud, axis=0), axis=0)
        else:
            raise ValueError
        ax.set_xlim(avg_pos[0] - .5 * axis_range, avg_pos[0] + .5 * axis_range)
        ax.set_ylim(avg_pos[1] - .5 * axis_range, avg_pos[1] + .5 * axis_range)
        ax.set_zlim(avg_pos[2] - .5 * axis_range, avg_pos[2] + .5 * axis_range)
    else:
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_zlim(-axis_limit, axis_limit)

    plt.draw()
    plt.pause(1e-3)


def show2d(point_cloud,
           camera=None,
           ax=None,
           c=None,
           image=None):
    """Plot point cloud and return subplot handle.

    Args:
        point_cloud: The point cloud as an array.
        camera: An Camera instance to project the point cloud to 2D points.
        ax: The matplotlib ax.
        c: Color of the point cloud.
        image: An image to be overlaid on.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax.cla()

    # Project.
    pixels = camera.project_point(point_cloud)

    # Create image.
    if image is None:
        image = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
    ax.imshow(image)

    # Plot.
    if len(point_cloud.shape) == 2:
        # Plot a holistic point cloud.
        if c is None:
            c = 'b'
        ax.scatter(pixels[:, 0], pixels[:, 1], s=1, c=c, alpha=0.8)
    elif len(point_cloud.shape) == 3:
        # Plot segmented point cloud.
        if c is None:
            c = COLOR_MAP
        for i in range(point_cloud.shape[0]):
            color = c[i]
            ax.scatter(pixels[i, :, 0], pixels[i, :, 1],
                       s=1, c=color, alpha=0.8)
    else:
        raise ValueError

    plt.draw()
    plt.pause(1e-3)
