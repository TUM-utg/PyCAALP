import os
import pickle

# import math
import csv
import numpy as np
import pyvista as pv
import trimesh
import fcl
import meshlib.mrmeshnumpy as mrmeshnumpy
import meshlib.mrmeshpy as mm


class DOFMatrixGenerator:

    def __init__(self):

        self.translational_range_factor = 2.0

        # How many translational values should be generated
        self.translational_distance_num = 5
        # How many rotational angles should be used
        self.rotational_angle_num = 6

        self.rotational_angles = np.array([15.0, 45.0, 75.0, 90.0, 120.0, 150.0])

        self.volume_precision = 0.1

    def detect_collision(self, moved_model1, model2_collision_object):
        """Detects collisions using the FCL library.

        :param moved_model1: The model1.
        :param model2_collision_object: The model2.
        :return: Whether the two models have collisions.
        """
        # moved_model1_trimesh = trimesh.Trimesh(
        #     moved_model1.points,
        #     moved_model1.faces.reshape((moved_model1.n_faces, 4))[:, 1:],
        # )
        # NOTE: changed for compatibility
        moved_model1_trimesh = trimesh.Trimesh(
            vertices=moved_model1.points,
            faces=moved_model1.faces.reshape((moved_model1.n_cells, 4))[:, 1:],
        )

        moved_model1_bvh = trimesh.collision.mesh_to_BVH(moved_model1_trimesh)

        moved_model1_collision_object = fcl.CollisionObject(
            moved_model1_bvh, fcl.Transform()
        )

        collision_request = fcl.CollisionRequest()
        collision_result = fcl.CollisionResult()

        collision_information = fcl.collide(
            moved_model1_collision_object,
            model2_collision_object,
            collision_request,
            collision_result,
        )

        return collision_result.is_collision

    def construct_dof_matrix_element(
        self, moved_model1, model2_collision_object, model2_dup
    ):
        """Constructs an entry in the DoF matrix.

        :param moved_model1: The model1.
        :param model2_collision_object: The model2 (in FCL data structure).
        :param model2_dup: The model2.
        :return: The value of the entry in the DoF matrix.
        """
        if self.detect_collision(moved_model1, model2_collision_object) is False:
            # If FCL result is False, then the predicted value is 1
            print("have no intersection.")
            return 1
        else:
            # Perform more detailed detection using MeshLib by computing the intersection volumes
            model1 = mrmeshnumpy.meshFromFacesVerts(
                moved_model1.faces.reshape((-1, 4))[:, 1:4], moved_model1.points
            )
            model2 = mrmeshnumpy.meshFromFacesVerts(
                model2_dup.faces.reshape((-1, 4))[:, 1:4], model2_dup.points
            )
            intersection_volume = mm.boolean(
                model1, model2, mm.BooleanOperation.Intersection
            ).mesh.volume()
            if intersection_volume > self.volume_precision:
                # If the intersection volume > pre-defined volume precision, then the predicted entry value is 0.
                print(f"have intersection. intersection volume: {intersection_volume}.")
                return 0
            else:
                # Otherwise, it is 1.
                print("have no intersection.")
                return 1

    def get_translational_values(self, model, joint_coordinate):
        """Gets the translational values based on the model.
        The vertices are projected on the x, y, z axis from joint_coordinate.
        The differences of the maximum and minimum coordinates of the projections are used as characteristic lengths.
        The characteristic lengths are multiplied by translational_range_factor, which are used as ranges.
        Translational values are generated with equal steps in the range.
        The number of translational values is translational_distance_num.

        :param model: The model.
        :param joint_coordinate: The joint coordinate matrix.
        :return: The generated translational values.
        """
        # Project the vertices on the axis
        projected_x = np.sum(joint_coordinate[1, :] * model.points, axis=1)
        projected_y = np.sum(joint_coordinate[2, :] * model.points, axis=1)
        projected_z = np.sum(joint_coordinate[3, :] * model.points, axis=1)

        # Compute the characteristic length
        x_range = (
            np.max(projected_x) - np.min(projected_x)
        ) * self.translational_range_factor
        y_range = (
            np.max(projected_y) - np.min(projected_y)
        ) * self.translational_range_factor
        z_range = (
            np.max(projected_z) - np.min(projected_z)
        ) * self.translational_range_factor

        x_distances = []
        y_distances = []
        z_distances = []

        # Generate the translational values
        for i in range(0, self.translational_distance_num):
            x_distances.append((i + 1) * x_range / self.translational_distance_num)
            y_distances.append((i + 1) * y_range / self.translational_distance_num)
            z_distances.append((i + 1) * z_range / self.translational_distance_num)

        translational_distances = {"x": x_distances, "y": y_distances, "z": z_distances}

        return translational_distances

    def process_joint_coordinate(self, joint_coordinate):
        """Processes the joint_coordinate matrix so that the orientations of x, y, z are unit vectors.

        :param joint_coordinate: The joint coordinate matrix.
        :return: The processed joint coordinate matrix.
        """
        for i in range(1, 4):
            axis_norm = np.sqrt(np.sum(joint_coordinate[i, :] * joint_coordinate[i, :]))
            joint_coordinate[i, :] = joint_coordinate[i, :] / axis_norm

        return joint_coordinate

    def construct_dof_matrix(self, model1_path, model2_path, joint_coordinate, part):
        """Constructs the DoF matrix of one model under joint_coordinate.
        The parameter part is either model1 or model2.
        If the parameter part is model1, the computed DoF matrix is for model1, otherwise, it is for model2

        :param model1_path: The file path of the model1.
        :param model2_path: The file path of the model2.
        :param joint_coordinate: The joint coordinate 4 by 3 matrix.
        :param part: Should compute the DoF matrix for model1 or model2.
        :return: The DoF matrix.
        """
        model1 = pv.read(model1_path)
        model2 = pv.read(model2_path)

        # joint_surface = pv.read(joint_surface_path)
        # joint_origin = self.compute_joint_coordinate_origin(joint_surface)

        dof_matrix = np.ones((3, 4), dtype=int)

        # Based on the parameter part, exchange the two models (this step is just to simplify the code)
        if part == "model1":
            model1_dup = model1.copy(deep=True)
            model2_dup = model2.copy(deep=True)
        else:
            model1_dup = model2.copy(deep=True)
            model2_dup = model1.copy(deep=True)

        # model2_trimesh = trimesh.Trimesh(
        #     model2_dup.points, model2_dup.faces.reshape((model2_dup.n_faces, 4))[:, 1:]
        # )
        # NOTE: changed it because of newer versions incompatibilitt
        model2_trimesh = trimesh.Trimesh(
            vertices=model2_dup.points,
            faces=model2_dup.faces.reshape((model2_dup.n_cells, 4))[:, 1:],
        )

        model2_bvh = trimesh.collision.mesh_to_BVH(model2_trimesh)

        model2_collision_object = fcl.CollisionObject(model2_bvh, fcl.Transform())

        # Ensure that the orientations of x, y ,z are unit vectors
        joint_coordinate = self.process_joint_coordinate(joint_coordinate)

        original_point = joint_coordinate[0, :]

        indicators = ["x", "y", "z"]

        # Get the translational values
        translational_values = self.get_translational_values(
            model1_dup, joint_coordinate
        )

        # From x axis to z axis
        for i in range(1, 4):

            orientation_vector = joint_coordinate[i, :]

            # Compute T+
            print(f"considering T" + indicators[i - 1] + "+")
            degree_of_freedom = 1
            for value in translational_values[indicators[i - 1]]:
                moved_model1 = model1_dup.copy(deep=True)
                moved_model1.translate(orientation_vector * value, inplace=True)
                if (
                    self.construct_dof_matrix_element(
                        moved_model1, model2_collision_object, model2_dup
                    )
                    == 0
                ):
                    degree_of_freedom = 0
                    break
            dof_matrix[i - 1, 0] = degree_of_freedom

            # Compute T-
            print(f"considering T" + indicators[i - 1] + "-")
            degree_of_freedom = 1
            for value in translational_values[indicators[i - 1]]:
                moved_model1 = model1_dup.copy(deep=True)
                moved_model1.translate(-orientation_vector * value, inplace=True)
                if (
                    self.construct_dof_matrix_element(
                        moved_model1, model2_collision_object, model2_dup
                    )
                    == 0
                ):
                    degree_of_freedom = 0
                    break
            dof_matrix[i - 1, 1] = degree_of_freedom

            # Compute R+
            print(f"considering R" + indicators[i - 1] + "+")
            degree_of_freedom = 1
            for j in range(0, self.rotational_angle_num):
                moved_model1 = model1_dup.copy(deep=True)
                moved_model1.rotate_vector(
                    orientation_vector,
                    self.rotational_angles[j],
                    original_point,
                    inplace=True,
                )
                if (
                    self.construct_dof_matrix_element(
                        moved_model1, model2_collision_object, model2_dup
                    )
                    == 0
                ):
                    degree_of_freedom = 0
                    break
            dof_matrix[i - 1, 2] = degree_of_freedom

            # Compute R-
            print(f"considering R" + indicators[i - 1] + "-")
            degree_of_freedom = 1
            for j in range(0, self.rotational_angle_num):
                moved_model1 = model1_dup.copy(deep=True)
                moved_model1.rotate_vector(
                    orientation_vector,
                    -self.rotational_angles[j],
                    original_point,
                    inplace=True,
                )
                if (
                    self.construct_dof_matrix_element(
                        moved_model1, model2_collision_object, model2_dup
                    )
                    == 0
                ):
                    degree_of_freedom = 0
                    break
            dof_matrix[i - 1, 3] = degree_of_freedom

        return dof_matrix


def create_pjinfo_and_jointcoords_lists(_data_dir):
    """Creates the part joint information and joint coordinates lists.

    :param _data_dir: The data directory (STL, csv).
    :return: The part joint information and joint coordinates lists.
    """
    _part_joint_information = []

    with open(_data_dir + "/stl/part_joint.csv", "r", encoding="utf-8") as _file:
        _csv_reader = csv.reader(_file)
        for _row in _csv_reader:
            if "Null" in _row or "NULL" in _row:
                continue
            _part_joint_information.append(_row)

    _joint_coordinates = []

    with open(_data_dir + "/stl/joint_coordinates.csv", "r", encoding="utf-8") as _file:
        _csv_reader = csv.reader(_file)
        for _row in _csv_reader:
            _joint_coordinates.append(_row)

    return _part_joint_information, _joint_coordinates


def calculate_joint_coordinate_origin(_joint_name, _joint_coordinates):
    """Calculates the joint coordinate origin based on the part joint information
    and joint coordinates.

    :param _joint_name: The joint name.
    :param _joint_coordinates: The joint coordinates.
    :return: The joint coordinate origin. It is a 4 by 3 matrix.
    """
    _joint_coordinate = np.zeros((12,))

    for _coordinate in _joint_coordinates:
        if _coordinate[0] == _joint_name:
            _index = 0
            while _index < 12:
                _joint_coordinate[_index] = float(_coordinate[_index + 1])
                _index += 1

    # joint_coordinate is a 4 by 3 matrix, the first row is the origin coordinate
    # the next three rows are the orientations of x, y, z
    _joint_coordinate = _joint_coordinate.reshape((4, 3))

    return _joint_coordinate


def create_freedom_matrix_dict(_data_dir, _part_joint_information, _joint_coordinates):
    """Creates a dictionary of freedom matrices. Keys are the part names and
    values as freedom matrices.

    :param _data_dir: The data directory (STL).
    :param _part_joint_information: The part joint information.
    :param _joint_coordinates: The joint coordinates.
    :return: The freedom matrix dictionary.
    """
    _dof_matrix_generator = DOFMatrixGenerator()

    _freedom_matrix_dict = {}
    for _part_joint in _part_joint_information:
        _model1_path = f"{_data_dir}/stl/{_part_joint[1]}.stl"
        _model2_path = f"{_data_dir}/stl/{_part_joint[2]}.stl"
        _joint_coordinate = calculate_joint_coordinate_origin(
            _part_joint[0], _joint_coordinates
        )

        _freedom_matrix_m1 = _dof_matrix_generator.construct_dof_matrix(
            _model1_path,
            _model2_path,
            _joint_coordinate,
            "model1",
        )

        _freedom_matrix_m2 = _dof_matrix_generator.construct_dof_matrix(
            _model1_path,
            _model2_path,
            _joint_coordinate,
            "model2",
        )

        _part_name = (_part_joint[1], _part_joint[2])
        _freedom_matrix_dict[_part_name] = [_freedom_matrix_m1, _freedom_matrix_m2]

    return _freedom_matrix_dict


def create_freedom_matrix_dict_complete(_data_dir):
    """Creates a dictionary of freedom matrices."""

    _part_joint_information, _joint_coordinates = create_pjinfo_and_jointcoords_lists(
        _data_dir
    )

    _dfm_dict = create_freedom_matrix_dict(
        _data_dir, _part_joint_information, _joint_coordinates
    )

    return _dfm_dict


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    DATA_DIR = "data/35up"

    dfm_dict = create_freedom_matrix_dict_complete(DATA_DIR)

    print(dfm_dict)
    pkl_dfm_name = "dfm_35_up.pkl"
    with open(pkl_dfm_name, "wb") as file:
        pickle.dump(dfm_dict, file)

    # dof_matrix_generator = DOFMatrixGenerator()

    # part_joint_information = []

    # data_dir = "test_data"

    # # Each row in part_joint.csv: joint_id, part1_id, part2_id
    # with open(
    #     data_dir + "/tabular_files/part_joint.csv", "r", encoding="utf-8"
    # ) as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         if "Null" in row or "NULL" in row:
    #             continue
    #         part_joint_information.append(row)

    # joint_coordinates = []

    # # Each row in joint_coordinates.csv: joint_id, origin coordinates (x0, y0, z0), orientations of x, y, z
    # with open(
    #     data_dir + "/tabular_files/joint_coordinates.csv", "r", encoding="utf-8"
    # ) as file:
    #     csv_reader = csv.reader(file)
    #     for row in csv_reader:
    #         joint_coordinates.append(row)

    # model1_path = f"test_data/stl_files/{part_joint_information[1][1]}.stl"
    # model2_path = f"test_data/stl_files/{part_joint_information[1][2]}.stl"

    # joint_coordinate = np.zeros((12,))

    # for coordinate in joint_coordinates:
    #     if coordinate[0] == part_joint_information[1][0]:
    #         index = 0
    #         while index < 12:
    #             joint_coordinate[index] = float(coordinate[index + 1])
    #             index += 1

    # joint_coordinate is a 4 by 3 matrix, the first row is the origin coordinate
    # the next three rows are the orientations of x, y, z
    # joint_coordinate = joint_coordinate.reshape((4, 3))

    # model1 = pv.read(model1_path)
    # model2 = pv.read(model2_path)
    # pl = pv.Plotter()
    # pl.add_mesh(model1, color="b")
    # pl.add_mesh(model2, color="g")
    # x_point = joint_coordinate[0] + 30 * joint_coordinate[1]
    # y_point = joint_coordinate[0] + 30 * joint_coordinate[2]
    # z_point = joint_coordinate[0] + 30 * joint_coordinate[3]
    # pl.add_points(joint_coordinate[0], color="y")
    # pl.add_lines(np.array([joint_coordinate[0], x_point]), color="r")
    # pl.add_lines(np.array([joint_coordinate[0], y_point]), color="g")
    # pl.add_lines(np.array([joint_coordinate[0], z_point]), color="b")
    # pl.show()

    # print(
    #     dof_matrix_generator.construct_dof_matrix(
    #         model1_path, model2_path, joint_coordinate, "model2"
    #     )
    # )
