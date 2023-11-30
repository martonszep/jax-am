import jax
import numpy as onp
import os
import meshio
import json
import yaml

import time
from functools import wraps

from jax_am import logger


def json_parse(json_filepath):
    with open(json_filepath) as f:
        args = json.load(f)
    json_formatted_str = json.dumps(args, indent=4)
    print(json_formatted_str)
    return args


def yaml_parse(yaml_filepath):
    with open(yaml_filepath) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
        print("YAML parameters:")
        # TODO: These are just default parameters
        print(yaml.dump(args, default_flow_style=False))
        print("These are default parameters")
    return args


def box_mesh(Nx, Ny, Nz, domain_x, domain_y, domain_z):
    dim = 3
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    z = onp.linspace(0, domain_z, Nz + 1)
    xv, yv, zv = onp.meshgrid(x, y, z, indexing='ij')
    points_xyz = onp.stack((xv, yv, zv), axis=dim)
    points = points_xyz.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xyz = points_inds.reshape(Nx + 1, Ny + 1, Nz + 1)
    inds1 = points_inds_xyz[:-1, :-1, :-1]
    inds2 = points_inds_xyz[1:, :-1, :-1]
    inds3 = points_inds_xyz[1:, 1:, :-1]
    inds4 = points_inds_xyz[:-1, 1:, :-1]
    inds5 = points_inds_xyz[:-1, :-1, 1:]
    inds6 = points_inds_xyz[1:, :-1, 1:]
    inds7 = points_inds_xyz[1:, 1:, 1:]
    inds8 = points_inds_xyz[:-1, 1:, 1:]
    cells = onp.stack((inds1, inds2, inds3, inds4, inds5, inds6, inds7, inds8),
                      axis=dim).reshape(-1, 8)
    out_mesh = meshio.Mesh(points=points, cells={'hexahedron': cells})
    return out_mesh


def rectangle_mesh(Nx, Ny, domain_x, domain_y, ele_type, periodic=False):
    """Create structured rectangle mesh."""
    dim = 2
    x = onp.linspace(0, domain_x, Nx + 1)
    y = onp.linspace(0, domain_y, Ny + 1)
    xv, yv = onp.meshgrid(x, y, indexing='ij')
    points_xy = onp.stack([xv, yv], axis=dim)
    points = points_xy.reshape(-1, dim)
    points_inds = onp.arange(len(points))
    points_inds_xy = points_inds.reshape(Nx + 1, Ny + 1)
    
    # if not periodic:
    #     inds1 = points_inds_xy[:-1, :-1]
    #     inds2 = points_inds_xy[1:, :-1]
    #     inds3 = points_inds_xy[1:, 1:]
    #     inds4 = points_inds_xy[:-1, 1:]
    # else:
    #     xv = onp.stack([onp.cos(2*onp.pi/domain_x*points[:,0]), onp.sin(2*onp.pi/domain_x*points[:,0])], axis=1)
    #     yv = onp.stack([onp.cos(2*onp.pi/domain_y*points[:,1]), onp.sin(2*onp.pi/domain_y*points[:,1])], axis=1)
    #     points = onp.concatenate([xv, yv], axis=1)

    #     inds1 = points_inds_xy
    #     inds2 = onp.roll(points_inds_xy, shift=-1, axis=0)
    #     inds3 = onp.roll(inds2, shift=-1, axis=1)
    #     inds4 = onp.roll(points_inds_xy, shift=-1, axis=1)

    if ele_type=='quad':
        inds1 = points_inds_xy[:-1, :-1]
        inds2 = points_inds_xy[1:, :-1]
        inds3 = points_inds_xy[1:, 1:]
        inds4 = points_inds_xy[:-1, 1:]
        cells = onp.stack((inds1, inds2, inds3, inds4), axis=dim).reshape(-1, 4)
        out_mesh = meshio.Mesh(points=points, cells={'quad': cells})
    elif ele_type=='triangle':
        inds1 = points_inds_xy[:-1, :-1]
        inds2 = points_inds_xy[1:, :-1]
        inds3 = points_inds_xy[1:, 1:]
        inds4 = points_inds_xy[:-1, 1:]
        cells1 = onp.stack((inds1, inds2, inds3), axis=dim).reshape(-1, 3)
        cells2 = onp.stack((inds1, inds3, inds4), axis=dim).reshape(-1, 3)
        cells = onp.concatenate([cells1, cells2])
        out_mesh = meshio.Mesh(points=points, cells={'triangle': cells})
    elif ele_type=='triangle6':
        assert Nx%2==0 and Ny%2==0, "Nx and Ny must be even for triangle6!"
        inds1 = points_inds_xy[:-2:2, :-2:2]
        inds2 = points_inds_xy[1:-1:2, :-2:2]
        inds3 = points_inds_xy[2::2, :-2:2]
        inds4 = points_inds_xy[:-2:2, 1:-1:2]
        inds5 = points_inds_xy[1:-1:2, 1:-1:2]
        inds6 = points_inds_xy[2::2, 1:-1:2]
        inds7 = points_inds_xy[:-2:2, 2::2]
        inds8 = points_inds_xy[1:-1:2, 2::2]
        inds9 = points_inds_xy[2::2, 2::2]
        cells1 = onp.stack((inds1, inds3, inds9, inds2, inds6,  inds5), axis=dim).reshape(-1, 6)
        cells2 = onp.stack((inds1, inds9, inds7, inds5, inds8,  inds4), axis=dim).reshape(-1, 6)
        cells = onp.concatenate([cells1, cells2])
        out_mesh = meshio.Mesh(points=points, cells={'triangle6': cells})
    else:
        raise NotImplementedError
    return out_mesh


def make_video(data_dir):
    # The command -pix_fmt yuv420p is to ensure preview of video on Mac OS is
    # enabled
    # https://apple.stackexchange.com/questions/166553/why-wont-video-from-ffmpeg-show-in-quicktime-imovie-or-quick-preview
    # The command -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" is to solve the following
    # "not-divisible-by-2" problem
    # https://stackoverflow.com/questions/20847674/ffmpeg-libx264-height-not-divisible-by-2
    # -y means always overwrite

    # TODO
    os.system(
        f'ffmpeg -y -framerate 10 -i {data_dir}/png/tmp/u.%04d.png -pix_fmt yuv420p -vf \
               "crop=trunc(iw/2)*2:trunc(ih/2)*2" {data_dir}/mp4/test.mp4') # noqa


# A simpler decorator for printing the timing results of a function
def timeit(func):

    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        logger.debug(f'Function {func.__name__} took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


# Wrapper for writing timing results to a file
def walltime(txt_dir=None, filename=None):

    def decorate(func):

        def wrapper(*list_args, **keyword_args):
            start_time = time.time()
            return_values = func(*list_args, **keyword_args)
            end_time = time.time()
            time_elapsed = end_time - start_time
            platform = jax.lib.xla_bridge.get_backend().platform
            logger.info(
                f"Time elapsed {time_elapsed} of function {func.__name__} "
                f"on platform {platform}"
            )
            if txt_dir is not None:
                os.makedirs(txt_dir, exist_ok=True)
                fname = 'walltime'
                if filename is not None:
                    fname = filename
                with open(os.path.join(txt_dir, f"{fname}_{platform}.txt"),
                          'w') as f:
                    f.write(f'{start_time}, {end_time}, {time_elapsed}\n')
            return return_values

        return wrapper

    return decorate
