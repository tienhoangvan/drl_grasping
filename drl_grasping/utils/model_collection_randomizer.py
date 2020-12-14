from typing import List
from networkx.algorithms.cuts import volume

import numpy as np

import os

from pcg_gazebo.parsers import parse_sdf
from pcg_gazebo.parsers.sdf import create_sdf_element


from scenario import gazebo as scenario_gazebo
from gym_ignition.utils import logger


import glob

import trimesh


# TODO: set RNG seed somehow

# Note: only models with mesh geometry are supported


class ModelCollectionRandomizer():

    _class_model_paths = None
    __sdf_base_name = 'model.sdf'
    __configured_sdf_base_name = 'model_modified.sdf'
    __blacklisted_base_name = 'BLACKLISTED'
    __collision_mesh_dir = 'meshes/collision/'
    __collision_mesh_file_type = 'stl'

    def __init__(self,
                 model_paths=None,
                 owner='GoogleResearch',
                 collection='Google Scanned Objects',
                 server='https://fuel.ignitionrobotics.org',
                 server_version='1.0',
                 unique_cache=False,
                 reset_collection=False,
                 enable_blacklisting=False):

        # If enabled, the newly created objects of this class will use its own individual cache
        # for model paths and must discover/download them on its own
        self._unique_cache = unique_cache

        # Flag that determines if models that cannot be used are blacklisted
        self._enable_blacklisting = enable_blacklisting

        # If enabled, the cache of the class used to store model paths among instances will be reset
        if reset_collection and not self._unique_cache:
            self._class_model_paths = None

        # Get file path to all models from
        # a) `model_paths` arg
        # b) local cache owner (if `owner` has some models, i.e `collection` is already downloaded)
        # c) Fuel collection (if `owner` has no models in local cache)
        if model_paths is not None:
            # Use arg
            if self._unique_cache:
                self._model_paths = model_paths
            else:
                self._class_model_paths = model_paths
        else:
            # Use local cache or Fuel
            if self._unique_cache:
                self._model_paths = self.get_collection_model_paths(owner=owner,
                                                                    collection=collection,
                                                                    server=server,
                                                                    server_version=server_version)
            elif self._class_model_paths is None:
                # Executed only once, unless the paths are reset with `reset_collection` arg
                self._class_model_paths = self.get_collection_model_paths(owner=owner,
                                                                          collection=collection,
                                                                          server=server,
                                                                          server_version=server_version)

    def get_collection_model_paths(self,
                                   owner='GoogleResearch',
                                   collection='Google Scanned Objects',
                                   server='https://fuel.ignitionrobotics.org',
                                   server_version='1.0'
                                   ) -> List[str]:

        # First check the local cache (for performance)
        # Note: This unfortunately does not check if models belong to the specified collection
        model_paths = scenario_gazebo.get_local_cache_model_paths(owner=owner,
                                                                  name='')
        if len(model_paths) > 0:
            return model_paths

        # Else download the models from Fuel and then try again
        collection_uri = '%s/%s/%s/collections/%s' % (server,
                                                      server_version,
                                                      owner,
                                                      collection)
        download_command = 'ign fuel download -v 3 -t model -j %s -u "%s"' % (os.cpu_count(),
                                                                              collection_uri)
        os.system(download_command)

        model_paths = scenario_gazebo.get_local_cache_model_paths(owner=owner,
                                                                  name='')
        if 0 == len(model_paths):
            logger.error('URI "%s" is not valid and does not contain any models that are \
                          owned by the owner of the collection' % collection_uri)
            pass

        return model_paths

    def get_random_model_path(self) -> str:

        if self._unique_cache:
            return np.random.choice(self._model_paths)
        else:
            return np.random.choice(self._class_model_paths)

    def random_model(self,
                     return_sdf_path=True,
                     max_faces=40000,
                     max_vertices=None,
                     skip_blacklisted=True,
                     min_density=500.0,
                     max_density=1500.0,
                     component_min_faces_fraction=0.05,
                     component_min_volume_fraction=0.1,
                     min_friction=0.75,
                     max_friction=1.5,
                     decimation_fraction_of_visual=0.025,
                     decimation_min_faces=8,
                     decimation_max_faces=500,
                     fix_mtl_texture_paths=True,
                     scale_std_dev=0.5,
                     min_scale=0.1,
                     max_scale=0.5,
                     ) -> str:

        # Loop until a model is found, checked for validity, configured and returned
        # If any of these steps fail, sample another model and try again
        # Note: Due to this behaviour, the function could stall if all models are invalid
        while True:

            # Get path to a random model from the collection
            model_path = self.get_random_model_path()

            # Check if the model is already blacklisted and skip if desired
            if skip_blacklisted and self.is_blacklisted(model_path):
                continue

            # Check is the model is already configured
            configured_sdf_path = self.get_configured_sdf_path(model_path)
            # TODO: enable
            # if os.path.isfile(configured_sdf_path):
            #     # If already configured, apply only randomization and return
            #     self.randomize_configured_model(configured_sdf_path,
            #                                     scale_std_dev=scale_std_dev,
            #                                     min_scale=min_scale,
            #                                     max_scale=max_scale,
            #                                     min_friction=min_friction,
            #                                     max_friction=max_friction,
            #                                     min_density=min_density,
            #                                     max_density=max_density,
            #                                     min_mass=0.1,
            #                                     max_mass=3.0)
            #     if return_sdf_path:
            #         return configured_sdf_path
            #     else:
            #         return model_path

            # Parse the SDF of the model
            sdf = parse_sdf(self.get_sdf_path(model_path))

            # Process the model(s) contained in the SDF
            for model in sdf.models:

                # Process the link(s) of each model
                for link in model.links:

                    # Get rid of the existing collisions prior to simplifying it
                    link.collisions.clear()

                    # Create default values for the total inertial properties of current link
                    # Their values will be updated for each body that the link contains
                    total_mass = 0.0
                    total_inertia = [[0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0],
                                     [0.0, 0.0, 0.0]]
                    common_centre_of_mass = [0.0, 0.0, 0.0]

                    # Go through the visuals and process them
                    for visual in link.visuals:

                        # Get path to the mesh of the link's visual
                        mesh_path = self.get_mesh_path(
                            model_path, visual)

                        # If desired, fix texture path in 'mtl' files for '.obj' mesh format
                        if fix_mtl_texture_paths:
                            self.fix_mtl_texture_paths(
                                model_path, mesh_path, model.attributes['name'])

                        # Load the mesh, without materials because we do not use them
                        mesh = trimesh.load(mesh_path,
                                            force='mesh',
                                            skip_materials=True)

                        # Blacklist models with too much geometry
                        if not self.check_excessive_geometry(mesh, model_path,
                                                             max_faces=max_faces,
                                                             max_vertices=max_vertices):
                            break

                        # Blacklist objects with disconnected geometry/components (multiple objects)
                        if not self.check_disconnected_components(mesh, model_path,
                                                                  component_min_faces_fraction=component_min_faces_fraction,
                                                                  component_min_volume_fraction=component_min_volume_fraction):
                            break

                        # Compute inertial properties for this mesh
                        total_mass, total_inertia, common_centre_of_mass = \
                            self.sum_inertial_properties(mesh,
                                                         total_mass, total_inertia, common_centre_of_mass,
                                                         min_density, max_density)

                        # Add decimated collision geometry to the SDF
                        self.add_collision(
                            mesh,
                            link,
                            model_path,
                            fraction_of_visual=decimation_fraction_of_visual,
                            min_faces=decimation_min_faces,
                            max_faces=decimation_max_faces,
                            min_friction=min_friction,
                            max_friction=max_friction)

                    else:
                        # Make sure the object has valid inertial properties, else blacklist
                        if not self.check_inertial_properties(model_path,
                                                              total_mass,
                                                              total_inertia):
                            break

                        # Write inertial properties to the SDF of the link
                        self.write_inertial_properties(link,
                                                       total_mass,
                                                       total_inertia,
                                                       common_centre_of_mass,
                                                       min_mass=0.1,
                                                       max_mass=3.0)

                        continue
                    break

                else:
                    continue
                break

            else:
                # Write the configured SDF into a file
                sdf.export_xml(configured_sdf_path)

                print(sdf)  # TODO: remove

                self.randomize_configured_model(configured_sdf_path)

                # Return path to the configured SDF file / model path
                if return_sdf_path:
                    return configured_sdf_path
                else:
                    return model_path

    def check_excessive_geometry(self,
                                 mesh,
                                 model_path,
                                 max_faces=None,
                                 max_vertices=None) -> bool:

        if max_faces is not None:
            num_faces = len(mesh.faces)
            if num_faces > max_faces:
                self.blacklist_model(model_path,
                                     reason='Excessive geometry (%d faces)' % num_faces)
                return False

        if max_vertices is not None:
            num_vertices = len(mesh.vertices)
            if num_vertices > max_vertices:
                self.blacklist_model(model_path,
                                     reason='Excessive geometry (%d vertices)' % num_vertices)
                return False

        return True

    def check_disconnected_components(self,
                                      mesh,
                                      model_path,
                                      component_min_faces_fraction=0.05,
                                      component_min_volume_fraction=0.1) -> bool:

        # Get a list of all connected componends inside the mesh
        # Consider components only with `component_min_faces_fraction`% faces
        min_faces = round(component_min_faces_fraction*len(mesh.faces))
        connected_components = trimesh.graph.connected_components(mesh.face_adjacency,
                                                                  min_len=min_faces)

        # If more than 1 objects were detected, consider also volume of the meshes
        if len(connected_components) > 1:
            total_volume = mesh.volume

            for component in connected_components:
                submesh = mesh.copy()
                mask = np.zeros(len(mesh.faces), dtype=np.bool)
                mask[component] = True
                submesh.update_faces(mask)

                volume_fraction = submesh.volume/total_volume
                if volume_fraction > component_min_volume_fraction:
                    self.blacklist_model(model_path,
                                         reason='Disconnected components \
                                         (%d instances, stopped at %f volume)' %
                                         (len(connected_components), volume_fraction))
                    return False

        return True

    def check_inertial_properties(self, model_path, mass, inertia) -> bool:

        if (
            mass < 1e-10 or
            inertia[0][0] < 1e-10 or
            inertia[1][1] < 1e-10 or
            inertia[2][2] < 1e-10
        ):
            self.blacklist_model(model_path,
                                 reason='Invalid inertial properties')
            return False

        return True

    def sum_inertial_properties(self, mesh,
                                total_mass, total_inertia, common_centre_of_mass,
                                min_density=500.0, max_density=1500.0):

        # Sample random density for the mesh
        mesh.density = np.random.uniform(min_density, max_density)

        # Temp variable to store the mass of all previous geometry
        mass_of_others = total_mass

        # For each additional mesh, simply add the mass and inertia
        total_mass += mesh.mass
        total_inertia += mesh.moment_inertia

        # Compute a common centre of mass between all previous geometry and the new mesh
        common_centre_of_mass = [
            mass_of_others *
            common_centre_of_mass[0] + mesh.mass*mesh.center_mass[0],
            mass_of_others *
            common_centre_of_mass[1] + mesh.mass*mesh.center_mass[1],
            mass_of_others *
            common_centre_of_mass[2] + mesh.mass*mesh.center_mass[2]
        ] / total_mass

        return total_mass, total_inertia, common_centre_of_mass

    def write_inertial_properties(self, link, mass, inertia, centre_of_mass, min_mass=0.1, max_mass=3.0):

        link.mass = mass

        # Contraint the mass within min/max mass
        mass_scale_factor = 1.0
        if link.mass.value > max_mass:
            mass_scale_factor = max_mass/link.mass.value
            link.mass = max_mass
        elif link.mass.value < min_mass:
            mass_scale_factor = min_mass/link.mass.value
            link.mass = min_mass

        link.inertia.ixx = inertia[0][0] * mass_scale_factor
        link.inertia.iyy = inertia[1][1] * mass_scale_factor
        link.inertia.izz = inertia[2][2] * mass_scale_factor
        link.inertia.ixy = inertia[0][1] * mass_scale_factor
        link.inertia.ixz = inertia[0][2] * mass_scale_factor
        link.inertia.iyz = inertia[1][2] * mass_scale_factor
        link.inertial.pose = [centre_of_mass[0], centre_of_mass[1], centre_of_mass[2],
                              0.0, 0.0, 0.0]

    def add_collision(self,
                      mesh,
                      link,
                      model_path,
                      fraction_of_visual=0.05,
                      min_faces=8,
                      max_faces=750,
                      min_friction=0.5,
                      max_friction=1.5):

        collision_name = link.attributes['name'] + \
            '_collision_' + str(len(link.collisions))
        collision_mesh_path = self.get_collision_mesh_path(model_path,
                                                           collision_name)

        # Simplify mesh via decimation
        face_count = \
            min(
                max(
                    fraction_of_visual*len(mesh.faces),
                    min_faces
                ),
                max_faces
            )
        collision_mesh = mesh.simplify_quadratic_decimation(face_count)

        # Export the collision mesh to the appropriate location
        os.makedirs(os.path.dirname(collision_mesh_path), exist_ok=True)
        collision_mesh.export(collision_mesh_path,
                              file_type=self.__collision_mesh_file_type)

        # Create collision SDF element
        collision = create_sdf_element('collision')

        # Add collision geometry to the SDF
        collision.geometry.mesh = create_sdf_element('mesh')
        collision.geometry.mesh.uri = os.path.relpath(
            collision_mesh_path, start=model_path)

        # Add surface friction to the SDF of collision
        collision.surface = create_sdf_element('surface')
        collision.surface.friction = create_sdf_element('friction', 'surface')
        collision.surface.friction.ode = create_sdf_element('ode', 'collision')
        collision.surface.friction.ode.mu = np.random.uniform(
            min_friction, max_friction)
        collision.surface.friction.ode.mu2 = collision.surface.friction.ode.mu

        # Add it to the SDF of the link
        collision_name = os.path.basename(collision_mesh_path).split('.')[0]
        link.add_collision(collision_name, collision)

    def get_collision_mesh_path(self, model_path, collision_name) -> str:

        return os.path.join(model_path,
                            self.__collision_mesh_dir,
                            collision_name + '.' + self.__collision_mesh_file_type)

    def get_sdf_path(self, model_path) -> str:

        return os.path.join(model_path, self.__sdf_base_name)

    def get_configured_sdf_path(self, model_path) -> str:

        return os.path.join(model_path, self.__configured_sdf_base_name)

    def get_blacklisted_path(self, model_path) -> str:

        return os.path.join(model_path, self.__blacklisted_base_name)

    def get_mesh_path(self, model_path, visual_or_collision) -> str:

        # TODO: This might need fixing for certain collections/models
        mesh_uri = visual_or_collision.geometry.mesh.uri.value
        return os.path.join(model_path, mesh_uri)

    def blacklist_model(self, model_path, reason='Unknown'):

        if self._enable_blacklisting:
            bl_file = open(self.get_blacklisted_path(model_path), 'w')
            bl_file.write(reason)
            bl_file.close()
        logger.warn("Skipping model '%s'. Reason: %s.%s" %
                    (model_path, reason, ' (BLACKLISTED)' if self._enable_blacklisting else ''))

    def is_blacklisted(self, model_path) -> bool:

        return os.path.isfile(self.get_blacklisted_path(model_path))

    def fix_mtl_texture_paths(self, model_path, mesh_path, model_name):

        # The `.obj` files use mtl
        if mesh_path.endswith('.obj'):

            # Find all textures located in the model path, used later to relative linking
            texture_files = glob.glob(os.path.join(
                model_path, '**', 'textures', '*.*'))

            # Find location ot mtl file, if any
            mtllib_file = None
            with open(mesh_path, 'r') as file:
                for line in file:
                    if 'mtllib' in line:
                        mtllib_file = line.split(
                            ' ')[-1].strip()
                        break

            if mtllib_file is not None:
                mtllib_file = os.path.join(
                    os.path.dirname(mesh_path), mtllib_file)

                fin = open(mtllib_file, "rt")
                data = fin.read()
                for line in data.splitlines():
                    if 'map_' in line:
                        # Find the name of the texture/map in the mtl
                        map_file = line.split(' ')[-1].strip()

                        # Find the first match of the texture/map file
                        for texture_file in texture_files:
                            if (os.path.basename(texture_file) == map_file or
                                    os.path.basename(texture_file) == os.path.basename(map_file)):

                                new_texture_file_name = texture_file.replace(
                                    map_file, model_name+map_file)

                                os.rename(texture_file, new_texture_file_name)

                                # Apply the correct relative path
                                data = data.replace(map_file, os.path.relpath(
                                    new_texture_file_name, start=os.path.dirname(mesh_path)))
                                break
                fin.close()

                # Write in the correct data
                fout = open(mtllib_file, "wt")
                fout.write(data)
                fout.close()

    def randomize_configured_model(self,
                                   configured_sdf_path,
                                   scale_std_dev=0.5,
                                   min_scale=0.05,
                                   max_scale=0.25,
                                   min_friction=0.5,
                                   max_friction=1.5,
                                   min_density=500.0,
                                   max_density=1500.0,
                                   min_mass=0.1,
                                   max_mass=3.0):

        # Parse the already configured SDF that needs to be randomized
        sdf = parse_sdf(configured_sdf_path)

        # Process the model(s) contained in the SDF
        for model in sdf.models:

            # Process the link(s) of each model
            for link in model.links:
                # Note: This script currently supports only scaling of links with single mesh geometry
                can_be_scaled = True if len(link.collisions) == 1 else False
                scale_factor = 1.0

                # Go through the collisions and process them
                for collision in link.collisions:
                    # Determine a random scale factor for the mesh to use
                    if can_be_scaled:
                        # Get path to the mesh of the link's collision
                        mesh_path = self.get_mesh_path(
                            os.path.dirname(configured_sdf_path), collision)

                        # Load the mesh, without materials because we do not use them
                        mesh = trimesh.load(mesh_path,
                                            force='mesh',
                                            skip_materials=True)

                        current_scale = collision.geometry.mesh.scale.value[0]

                        mesh_scale = mesh.scale
                        scale_factor = np.random.uniform(current_scale*min_scale/mesh_scale,
                                                         current_scale*max_scale/mesh_scale)

                        collision.geometry.mesh.scale = [scale_factor] * 3

                    # Randomize friction
                    collision.surface.friction.ode.mu = np.random.uniform(
                        min_friction, max_friction)
                    collision.surface.friction.ode.mu2 = collision.surface.friction.ode.mu

                # Similarly go through the visual of the list
                for visual in link.visuals:
                    # Apply scale
                    visual.geometry.mesh.scale = [scale_factor] * 3

                # Recompute inertial properties acording to the scale
                link.inertial.pose.x *= scale_factor
                link.inertial.pose.y *= scale_factor
                link.inertial.pose.z *= scale_factor

                link.mass = link.mass.value * scale_factor**3

                # Constraint the mass to mix/max values
                mass_scale_factor = 1.0
                if link.mass.value > max_mass:
                    mass_scale_factor = max_mass/link.mass.value
                    link.mass = max_mass
                elif link.mass.value < min_mass:
                    mass_scale_factor = min_mass/link.mass.value
                    link.mass = min_mass

                scale_factor_n5 = scale_factor ** 5
                link.inertia.ixx = link.inertia.ixx.value * scale_factor_n5 * mass_scale_factor
                link.inertia.iyy = link.inertia.iyy.value * scale_factor_n5 * mass_scale_factor
                link.inertia.izz = link.inertia.izz.value * scale_factor_n5 * mass_scale_factor
                link.inertia.ixy = link.inertia.ixy.value * scale_factor_n5 * mass_scale_factor
                link.inertia.ixz = link.inertia.ixz.value * scale_factor_n5 * mass_scale_factor
                link.inertia.iyz = link.inertia.iyz.value * scale_factor_n5 * mass_scale_factor

        # (random) scale uniformly all <mesh><scale></scale><mesh> fields
        # mass multiply by 'scale**3'
        # inertia multiply by 'scale**5'
        # pose of inertia multiply by 'scale'
        # (random) randomize friction coefficients

        print(sdf)  # TODO: remove

        sdf.export_xml(configured_sdf_path)
