#!/usr/bin/env python

"""Process an OBJ file and create the URDF file.
"""

import argparse
import glob
import os

import numpy as np

import _init_paths  # NOQA
from robovat.utils import mesh_utils
from robovat.utils.logging import logger


# TODO(kuanfang): Temporally remove the V-HACD parameters for simplicity, will
# add them back if we find the results are not as good.
VHACD_COMMAND = (
    './bin/vhacd'
    ' --input {input_path:s}'
    ' --output {output_path:s}'
    # ' --concavity 0.0025'
    ' --log {log_path:s}'
    )


MESHCONV_COMMAND = (
    './bin/meshconv -c obj -tri '
    ' -o {output_path:s}'
    ' {input_path:s}'
    )


def parse_args():
    """Parse arguments.

    Returns:
        args: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description='Convert an obj 3D Description file '
                    'into a urdf file with collision convex meshes.')

    parser.add_argument('--input',
                        dest='input_pattern',
                        help='The pattern or path of the input model(s).',
                        type=str,
                        required=True)

    parser.add_argument('--output',
                        dest='output_dir',
                        help='The directory of the output model(s).',
                        type=str,
                        default=None)

    parser.add_argument('--decompose',
                        dest='decompose',
                        help='If True, decompose the object using VHACD.',
                        type=int,
                        default=0)

    parser.add_argument('--rgba',
                        dest='rgba',
                        help='The color of the object.',
                        default=None,
                        type=str)

    parser.add_argument('--scale',
                        dest='scale',
                        help='The scale of the object.',
                        default=1.0,
                        type=float)

    parser.add_argument('--mass',
                        dest='mass',
                        help='The mass of the object.',
                        default=0.1,
                        type=float)

    parser.add_argument('--density',
                        dest='density',
                        help='The density of the object.',
                        default=None,
                        type=float)

    args = parser.parse_args()

    return args


def count_output_groups(wrl_path):
    """Count the number of output groups.

    Args:
        wrl_path: Path to the WRL file.

    Returns:
        Number of groups.
    """
    num_groups = 0

    with open(wrl_path, 'r') as f:
        for line in f:
            if line.startswith('Group'):
                num_groups += 1

    return num_groups


def run_convex_decomposition(input_path,
                             output_path='output.wrl',
                             log_path='log.txt'):
    """Run the convex decomposition with V-HACD.

    Args:
        input_path: Path to the input OBJ file.
        output_path: Path to the output WRL file.
        log_path: Path to the log file.
    """
    # TODO(kuanfang): Temporally remove the constraints of number of groups for
    # simplicity, will add it back if we find it necessary in the future.
    command = VHACD_COMMAND.format(
        input_path=input_path,
        output_path=output_path,
        log_path=log_path)
    logger.info(command)
    os.system(command)


def split_wrl_file(input_path, tmp_dir='/tmp'):
    """Split the WRL file into pieces.

    Args:
        input_path: Path to the input WRL file.
        tmp_dir: A directory for saving the temporary files.

    Returns:
        output_paths: List of paths to the output WRL files.
    """
    with open(input_path, 'r') as f:
        data = f.read()
        data = data.splitlines()
        i = 0
        output_paths = []

        while i < len(data):
            num_pieces = len(output_paths)
            filename = 'tmp_vhacd_%d.wrl' % (num_pieces)
            output_path = os.path.join(tmp_dir, filename)
            output_paths.append(output_path)

            with open(output_path, 'w') as new_file:
                new_file.write(data[i])
                i = i + 1

                while (i < len(data)) and (data[i][:5] != '#VRML'):
                    new_file.write(data[i] + '\n')
                    i = i + 1

    return output_paths


def convert_wrl_to_obj(wrl_paths, output_dir, body_name=''):
    """Convert a WRL files into .obj files using meshconv.

    Args:
        wrl_paths: List of paths to the input WRL files.
        output_dir: Output directory.
        body_name: Name of the body.

    Returns:
        obj_filenames: List of filenames of the output OBJ files.
    """
    num_pieces = len(wrl_paths)
    obj_filenames = []

    for i, wrl_path in enumerate(wrl_paths):
        basename = '%s_vhacd_%d_of_%d' % (body_name, i, num_pieces)
        filename = '%s.obj' % (basename)
        obj_filenames.append(filename)
        obj_path = os.path.join(output_dir, basename)

        command = MESHCONV_COMMAND.format(
            input_path=wrl_path, output_path=obj_path)
        logger.info(command)
        os.system(command)

    return obj_filenames


def random_rgba(a=1.0):
    """Generate random rgba value.

    Args:
        a: Alpha value.

    Returns:
        A string of RGBA values to be written in URDF files.
    """
    r = np.random.uniform(0.0, 1.0)
    g = np.random.uniform(0.0, 1.0)
    b = np.random.uniform(0.0, 1.0)

    if a is None:
        a = np.random.uniform(0.0, 1.0)

    return '%.2f %.2f %.2f %.2f' % (r, g, b, a)


def process_object(input_path,
                   output_dir,
                   rgba,
                   scale=1.0,
                   mass=0.1,
                   density=None):
    """Process a single object.

    Args:
        input_path: The path to the input .obj file.
        output_dir: The directory of the processed model, including the .urdf
            file, and the decomposed .obj files.
        rgba: The color of the visual model.
        scale: The scale of the model.
        mass: The mass of the whole object.
        density: The density of the whole object.
    """
    body_name = os.path.splitext(os.path.basename(input_path))[0]

    # Copy the mesh(visual model) to the output directory (with the URDF file).
    # logger.info('Copying the mesh from %s to %s...'
    #             % (input_path, output_dir))
    # command = 'cp %s %s' % (input_path, output_dir)
    # logger.info(command)
    # os.system(command)

    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    output_dir = os.path.join(output_dir, body_name)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    tmp_dir = os.path.join(output_dir, 'tmp')
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)

    tmp_output_path = os.path.join(tmp_dir, 'output.wrl')

    # Run V-HACD to generate the collision meshes (as .wrl files).
    logger.info('Running the V-HACD convex decomposition...')
    log_path = os.path.join('./meshes', 'log.txt')
    run_convex_decomposition(input_path=input_path,
                             output_path=tmp_output_path,
                             log_path=log_path)

    # Split the V-HACD results (as the .wrl file) into pieces.
    logger.info('Splitting the V-HACD results at: %s...' % (tmp_output_path))
    wrl_paths = split_wrl_file(tmp_output_path, tmp_dir=tmp_dir)

    # Convert each .wrl file (each piece of V-HACD results) into .obj file.
    logger.info('Converting the split V-HACD results to .obj format.')
    vhacd_filenames = convert_wrl_to_obj(wrl_paths,
                                         output_dir,
                                         body_name)

    # Read the mesh and compute the mesh properties.
    logger.info('Reading the obj file at: %s...' % (input_path))
    vertices, triangles = mesh_utils.read_from_obj(input_path)
    centroid = mesh_utils.compute_centroid(vertices, triangles)
    volume = mesh_utils.compute_volume(vertices, triangles)
    surface_area = mesh_utils.compute_surface_area(vertices, triangles)

    if mass is None:
        raise ValueError('The volume is problematic. Do not use the density.')
        mass = volume * density
        # mass = surface_area * args.density
    else:
        mass = mass

    logger.info('Object Information: \n'
                '\tbody_name: %s\n'
                '\tcentroid: %s\n'
                '\tvolume: %f\n'
                '\tsurface_area: %f\n'
                '\tmass: %f\n'
                % (body_name, centroid, volume, surface_area, mass))

    # Create the URDF file.
    urdf_path = os.path.join(output_dir, body_name + '.urdf')
    logger.info('Writing the URDF files at: %s...' % (urdf_path))

    # Create the visual meshes.
    with open('./tools/templates/visual_template.xml', 'r') as f:
        visual_template = f.read()

    visual_text = ''
    for vhacd_filename in vhacd_filenames:
        visual_text += visual_template.format(
                filename=vhacd_filename,
                scale=scale)

    # Create the collision meshes.
    with open('./tools/templates/collision_template.xml', 'r') as f:
        collision_template = f.read()

    collision_text = ''
    for vhacd_filename in vhacd_filenames:
        collision_text += collision_template.format(
            filename=vhacd_filename,
            scale=scale)

    if rgba is None:
        rgba = random_rgba()

    # Write to the URDF file.
    with open('./tools/templates/urdf_template.xml', 'r') as f:
        urdf_template = f.read()

    with open(urdf_path, 'w') as f:
        urdf_text = urdf_template.format(
            body_name=body_name,
            mass=mass,
            ixx=1,
            iyy=1,
            izz=1,
            ixy=0,
            ixz=0,
            iyz=0,
            cx=centroid[0],
            cy=centroid[1],
            cz=centroid[2],
            visual=visual_text,
            collision=collision_text,
            rgba=rgba)

        f.write(urdf_text)

    # Clean up the temporal directory.
    logger.info('Removing the tmp folder...')
    command = 'rm -rf %s' % tmp_dir
    logger.info(command)
    os.system(command)


def create_urdf(input_path, output_dir, rgba, scale=1.0, mass=0.1):
    """Process a single object.

    Args:
        input_path: The path to the input .obj file.
        output_dir: The directory of the processed model, including the .urdf
            file, and the decomposed .obj files.
        rgba: The color of the visual model.
        scale: The scale of the model.
        mass: The mass of the whole object.
    """
    body_name = os.path.splitext(os.path.basename(input_path))[0]
    body_path = '%s.obj' % (body_name)

    # Read the mesh and compute the mesh properties.
    logger.info('Reading the obj file at: %s...' % (input_path))
    vertices, triangles = mesh_utils.read_from_obj(input_path)
    centroid = mesh_utils.compute_centroid(vertices, triangles)
    volume = mesh_utils.compute_volume(vertices, triangles)
    surface_area = mesh_utils.compute_surface_area(vertices, triangles)

    logger.info('Object Information: \n'
                '\tbody_name: %s\n'
                '\tcentroid: %s\n'
                '\tvolume: %f\n'
                '\tsurface_area: %f\n'
                '\tmass: %f\n'
                % (body_name, centroid, volume, surface_area, mass))

    # Create the URDF file.
    if output_dir is None:
        output_dir = os.path.dirname(input_path)

    urdf_path = os.path.join(output_dir, body_name + '.urdf')
    logger.info('Writing the URDF files at: %s...' % (urdf_path))

    # Create the visual meshes.
    with open('./tools/templates/visual_template.xml', 'r') as f:
        visual_template = f.read()
        visual_text = visual_template.format(
            filename=body_path,
            scale=scale)

    # Create the collision meshes.
    with open('./tools/templates/collision_template.xml', 'r') as f:
        collision_template = f.read()
        collision_text = collision_template.format(
            filename=body_path,
            scale=scale)

    if rgba is None:
        rgba = random_rgba()

    # Write to the URDF file.
    with open('./tools/templates/urdf_template.xml', 'r') as f:
        urdf_template = f.read()

    with open(urdf_path, 'w') as f:
        urdf_text = urdf_template.format(
            body_name=body_name,
            mass=mass,
            ixx=1,
            iyy=1,
            izz=1,
            ixy=0,
            ixz=0,
            iyz=0,
            cx=centroid[0],
            cy=centroid[1],
            cz=centroid[2],
            visual=visual_text,
            collision=collision_text,
            rgba=rgba)

        f.write(urdf_text)


def main():
    args = parse_args()

    if args.input_pattern[-4:] != '.obj':
        input_pattern = '%s.obj' % (args.input_pattern)
    else:
        input_pattern = args.input_pattern

    input_paths = glob.glob(input_pattern)
    logger.info('Found %d .obj files at %s.', len(input_paths), input_pattern)

    for i, input_path in enumerate(input_paths):
        logger.info('Processing object %d / %d.', i, len(input_paths))

        if args.decompose:
            process_object(input_path,
                           output_dir=args.output_dir,
                           rgba=args.rgba,
                           scale=args.scale,
                           mass=args.mass,
                           density=args.density)
        else:
            create_urdf(input_path,
                        output_dir=args.output_dir,
                        rgba=args.rgba,
                        scale=args.scale,
                        mass=args.mass)


if __name__ == '__main__':
    main()
