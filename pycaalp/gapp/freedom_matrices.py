"""All operations for the deegree freedom matrices."""

import math
import pandas as pd
import numpy as np
import networkx as nx


def filter_part_name(part_name: str) -> str:
    """Filter the part name to remove any special characters

    Args:
        part_name: The part name to be filtered.

    Returns:
        The filtered part name
    """
    if part_name.endswith("\xa0"):
        part_name = part_name.replace("\xa0", "")
    return part_name


def read_freedom_matrices(filename: str) -> dict:
    """x-	x+	y-	y+	z-	z+	rx-	rx+	ry-	ry+	rz-	rz+
    Freedom matrix dims: 4x3
    [ T+[1x3]; T-[1x3]; R+[1x3]; R-[1x3] ]

    Args:
        filename: The filename of the freedom matrix file

    Returns:
        A dictionary with the freedom matrix data
    """
    if filename.endswith(".csv"):
        fm_file = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        fm_file = pd.read_excel(filename, header=None)
    else:
        raise ValueError(f"Unsupported filetype: {filename}")

    fm_dict = {}
    for index, row in fm_file.iterrows():
        if index % 2 == 0:
            part_a = filter_part_name(row[1])
            part_b = filter_part_name(row[14])

            fm_dict[(part_a, part_b)] = []
        else:
            fm_part_a = np.array(
                [
                    [row[3], row[5], row[7]],
                    [row[2], row[4], row[6]],
                    [row[9], row[11], row[13]],
                    [row[8], row[10], row[12]],
                ],
                dtype=int,
            )
            fm_part_b = np.array(
                [
                    [row[16], row[18], row[20]],
                    [row[15], row[17], row[19]],
                    [row[22], row[24], row[26]],
                    [row[21], row[23], row[25]],
                ],
                dtype=int,
            )
            fm_dict[(part_a, part_b)] = [fm_part_a, fm_part_b]
    return fm_dict


def read_coord_sys(filename: str) -> dict:
    """Read the coordinate system origins and orientations of joints
    Matrix dims: 4x3
    [ origin; x-axis; y-axis; z-axis ]

    Args:
        filename: The filename of the coordinate system file

    Returns:
        A dictionary with the coordinate system data
    """
    if filename.endswith(".csv"):
        cs_file = pd.read_csv(filename)
    elif filename.endswith(".xlsx"):
        cs_file = pd.read_excel(filename, header=None)
    else:
        raise ValueError(f"Unsupported filetype: {filename}")

    cs_dict = {}
    for index, row in cs_file.iterrows():
        if index != 0:
            part_a = filter_part_name(row[1])
            part_b = filter_part_name(row[2])
            cs_dict[(part_a, part_b)] = np.array(row[3:], dtype=np.float16).reshape(
                4, 3
            )
    return cs_dict


def assign_fm_cs_to_main_graph(main_graph: nx.Graph, dfm_dict: dict) -> nx.Graph:
    """Assign the freedom matrices and coordinate systems to the main graph

    x-	x+	y-	y+	z-	z+	rx-	rx+	ry-	ry+	rz-	rz+
    Freedom matrix dims: 4x3
    [ T+[1x3]; T-[1x3]; R+[1x3]; R-[1x3] ]

    Args:
        main_graph: The main graph to which the freedom matrices and
                coordinate systems will be assigned
        dfm_dict: A dictionary with the freedom matrix and coordinate system data

    Returns:
        The main graph with the freedom matrices and coordinate systems assigned
    """
    for p1, p2, attr in main_graph.edges.data():
        edge = (p1, p2)
        curr_joint = attr["name"]
        main_graph.edges[edge]["fm_parta"] = np.array(
            dfm_dict[curr_joint]["dfm"][p1]
        ).transpose()
        main_graph.edges[edge]["fm_partb"] = np.array(
            dfm_dict[curr_joint]["dfm"][p2]
        ).transpose()
        main_graph.edges[edge]["coord_sys"] = np.array(
            [
                dfm_dict[curr_joint]["Origin"],
                dfm_dict[curr_joint]["Xuvec"],
                dfm_dict[curr_joint]["Yuvec"],
                dfm_dict[curr_joint]["Zuvec"],
            ]
        )

    return main_graph


class Trafo:
    """A class to represent the transformation matrix and rotation matrices of a joint"""

    def __init__(self, cs_mat: np.ndarray):
        """Initialize the transformation matrix and rotation matrices of a joint

        Args:
            cs_mat: The coordinate system matrix
        """
        assert isinstance(cs_mat, np.ndarray), "cs_mat should be np.ndarray"
        assert cs_mat.shape == (4, 3), "cs_mat shape should be (4,3)"

        self.trans_mat = None
        self.angles = None
        self.rot_x_mat = None
        self.rot_y_mat = None
        self.rot_z_mat = None
        self.total_transf_mat = None

        # Calculations
        self.trans_mat = np.concatenate(
            (
                np.concatenate((np.identity(3), cs_mat[0, :].reshape(3, 1)), axis=1),
                np.array([[0, 0, 0, 1]]),
            ),
            axis=0,
        )
        self.angles = self.calculate_angles(cs_mat)
        self.calculate_rot_mats()
        self.calculate_total_transf_mat()

    def calculate_angles(self, cs_mat: np.ndarray) -> tuple[float, float, float]:
        """Calculate the angles of the unit vector to be used in the rotation matrix
        with respect to the x, y, and z axes

        Args:
            cs_mat: The coordinate system matrix

        Returns:
            A tuple with the angles (in rads) of the unit vectors
        """
        # A: access ux
        # B: access vy
        # C: access wz
        _a = math.acos(cs_mat[1, 0])
        _b = math.acos(cs_mat[2, 1])
        _c = math.acos(cs_mat[3, 2])
        return _a, _b, _c

    def calculate_rot_mats(self):
        """Calculate the rotation matrices for the x, y, and z axes
        based on the angles calculated from the unit vectors

        Updates rot_x_mat, rot_y_mat, rot_z_mat attributes.
        """
        self.rot_x_mat = np.array(
            [
                [1, 0, 0, 0],
                [0, math.cos(self.angles[0]), -math.sin(self.angles[0]), 0],
                [0, math.sin(self.angles[0]), math.cos(self.angles[0]), 0],
                [0, 0, 0, 1],
            ]
        )

        self.rot_y_mat = np.array(
            [
                [math.cos(self.angles[1]), 0, math.sin(self.angles[1]), 0],
                [0, 1, 0, 0],
                [-math.sin(self.angles[1]), 0, math.cos(self.angles[1]), 0],
                [0, 0, 0, 1],
            ]
        )

        self.rot_z_mat = np.array(
            [
                [math.cos(self.angles[2]), -math.sin(self.angles[2]), 0, 0],
                [math.sin(self.angles[2]), math.cos(self.angles[2]), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def calculate_total_transf_mat(self):
        """Calculate the total transformation matrix based on the rotation matrices
        and the translation matrix
        """
        self.total_transf_mat = (
            self.trans_mat @ self.rot_x_mat @ self.rot_y_mat @ self.rot_z_mat
        )


def init_trafo_mats(main_graph: nx.Graph) -> nx.Graph:
    """Initialize the transformation matrices for the main graph

    Args:
        main_graph: The main graph to which the transformation matrices will be assigned

    Returns:
        The main graph with the transformation matrices assigned per joint
    """
    for p1, p2 in main_graph.edges():
        # Initialize trafo class
        main_graph[p1][p2]["trafo"] = Trafo(main_graph[p1][p2]["coord_sys"])
    return main_graph


def calculate_from_to_transf_mat(from_trafo: Trafo, to_trafo: Trafo) -> np.ndarray:
    """Calculate the transformation matrix from one joint to another
    from_total_trans * inv(to_total_trans)

    Args:
        from_trafo: The transformation matrix of the source joint
        to_trafo: The transformation matrix of the target joint

    Returns:
        The transformation matrix from one joint to another
    """
    total_transf_mat = from_trafo.total_transf_mat @ np.linalg.inv(
        to_trafo.total_transf_mat
    )
    return total_transf_mat


def transform_dfm_to_ref_coords(
    dfm: np.ndarray, total_transfm_mat: np.ndarray
) -> np.ndarray:
    """Transform the DFM to the reference coordinates and return only the first two rows [3x2]

    Args:
        dfm: The DFM matrix
        total_transfm_mat: The total transformation matrix of the reference coordinates

    Returns:
        The transformed DFM matrix to the reference coordinates
    """
    # Step 1: Multiply the DFM matrix with the total transformation matrix
    transf_dfm = (
        np.concatenate((np.transpose(dfm), np.zeros((1, 4))), axis=0)
        @ total_transfm_mat
    )
    # Step 2: Use the absolute values of the matrix
    transf_dfm = np.abs(transf_dfm)
    # Step 3: Set the values of the matrix to 1 if they are greater than 0
    transf_dfm = np.where(transf_dfm > 1e-15, 1, 0)
    return transf_dfm[:3, :2]


def compare_non_zero_elements_in_rows(
    row_ref: np.ndarray, row_transf: np.ndarray
) -> bool:
    """Compare the non-zero elements in the rows of the reference and transformed DFM

    Args:
        row_ref: The row of the reference DFM
        row_tranf: The row of the transformed DFM

    Returns:
        True if non-zero elements of row_ref are greater or equal to row_transf,
        False otherwise
    """
    num_nz_ref = len(np.nonzero(row_ref)[0])
    num_nz_transf = len(np.nonzero(row_transf)[0])
    return num_nz_ref >= num_nz_transf


def make_collision_check(ref_dfm: np.ndarray, transf_dfm: np.ndarray) -> bool:
    """Make a collision check between the reference and transformed DFM

    Args:
        ref_dfm: The reference DFM matrix
        transf_dfm: The transformed DFM matrix

    Returns:
        True if there is a collision, False otherwise

    """
    return not compare_non_zero_elements_in_rows(
        ref_dfm[0], transf_dfm[:, 0]
    ) and not compare_non_zero_elements_in_rows(ref_dfm[1], transf_dfm[:, 0])


def complete_collision_check(
    main_graph: nx.Graph, current_graph: nx.Graph, new_edge: int, new_part: int
) -> bool:
    """Make a complete collision check between the reference and transformed DFM
    for all the edges connected to the new part in the main graph and their corresponding parts
    which exist in the current graph of the digraph.

    Args:
        main_graph: The initial assembly graph
        current_graph: The current graph in the assembly digraph.
        new_edge: New edge added to make the current_graph, at a specific digraph layer.
        new_part: New part added to make the current_graph, at a specific digraph layer.

    Returns:
        True if there a collision is detected, i.e., infeasible state.
        False otherwise, i.e., possible assembly state.
    """
    # Access the reference trafo and the DFM of the new joint/part
    ref_trafo = main_graph.edges[new_edge]["trafo"]
    if not ref_trafo:
        return False

    new_part = (new_part, 0) if new_edge[0] == new_part else (new_part, 1)
    dfm_ref = (
        main_graph.edges[new_edge]["fm_parta"]
        if new_part[1] == 0
        else main_graph.edges[new_edge]["fm_partb"]
    )

    # Iterate over all the possible connected edges to the new part
    # Exclude the new edge
    for edge in main_graph.edges(new_part[0]):
        if edge in [new_edge, (new_edge[1], new_edge[0])]:
            continue

        # Check if the other part of the edge (not the new part) is connected to the current graph
        other_part = (edge[1], 1) if edge[0] == new_part[0] else (edge[0], 0)
        if current_graph.edges(other_part[0]):

            # Access the trafo of the current edge to be used for the transformation
            from_trafo = main_graph.edges[edge]["trafo"]
            total_transf_mat = calculate_from_to_transf_mat(from_trafo, ref_trafo)
            # Access the DFM of the new_part for the current edge, not the other part
            dfm = (
                main_graph.edges[edge]["fm_parta"]
                if other_part[1] == 1
                else main_graph.edges[edge]["fm_partb"]
            )
            # Transform the DFM to the reference coordinates, i.e., the ones of the new_edge
            transf_dfm = transform_dfm_to_ref_coords(dfm, total_transf_mat)

            if make_collision_check(dfm_ref, transf_dfm):
                return True
    return False
