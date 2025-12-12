import os
import math
import csv
import numpy as np
import pyvista as pv
import trimesh
import fcl


class RelationshipClassifier:

    def __init__(self):

        # The spatial length of the displacements
        self.distance_precision = 0.01
        # The number of pre-defined displacements
        self.displacement_vector_num = 14

        sqrt_3 = math.sqrt(3)

        self.displacement_vectors = np.array(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, -1.0],
                [1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
                [-1.0 / sqrt_3, -1.0 / sqrt_3, -1.0 / sqrt_3],
                [1.0 / sqrt_3, 1.0 / sqrt_3, -1.0 / sqrt_3],
                [-1.0 / sqrt_3, -1.0 / sqrt_3, 1.0 / sqrt_3],
                [-1.0 / sqrt_3, 1.0 / sqrt_3, 1.0 / sqrt_3],
                [1.0 / sqrt_3, -1.0 / sqrt_3, -1.0 / sqrt_3],
                [-1.0 / sqrt_3, 1.0 / sqrt_3, -1.0 / sqrt_3],
                [1.0 / sqrt_3, -1.0 / sqrt_3, 1.0 / sqrt_3],
            ]
        )

        self.displacement_vectors *= self.distance_precision

        self.limit_distance = 50000.0

        self.file_folder_path = None
        self.file_names = None
        self.file_num = 0

        self.relationship_matrix = None

    def detect_collision(self, moved_model1, model2_collision_object):
        """Detects collisions using the FCL library.

        :param moved_model1: The model1.
        :param model2_collision_object: The model2.
        :return: Whether the two models have collisions.
        """

        moved_model1_trimesh = trimesh.Trimesh(
            moved_model1.points,
            moved_model1.faces.reshape((moved_model1.n_faces, 4))[:, 1:],
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

    def classify_contact_relationship(self, model1, model2):
        """Classifies if model1 and model2 have contact relationship.
        The classification is based on FCL library.

        :param model1: The model1.
        :param model2: The model2.
        :return: A Boolean value to indicate whether the two models have contact relationship.
        """
        # These data structure conversions are necessary
        model2_trimesh = trimesh.Trimesh(
            model2.points, model2.faces.reshape((model2.n_faces, 4))[:, 1:]
        )

        model2_bvh = trimesh.collision.mesh_to_BVH(model2_trimesh)

        model2_collision_object = fcl.CollisionObject(model2_bvh, fcl.Transform())

        have_contact = False

        model1_dup = model1.copy(deep=True)

        if self.detect_collision(model1_dup, model2_collision_object) is True:
            # If it is directly classified as having contact relationship, then have_contact is set to True
            have_contact = True
        else:
            # Otherwise, it still does not necessarily mean that the two models do not have contact relationship
            # because of some representation inaccuracies
            # Therefore, some pre-defined displacements are performed to model1
            # It will classify whether the moved model1 and the model2 have contact relationship
            for i in range(0, self.displacement_vector_num):

                moved_model1 = model1.copy(deep=True)
                moved_model1.translate(self.displacement_vectors[i, :], inplace=True)

                # If at least one moved model1 has contact relationship with model2, the classification is True
                if self.detect_collision(moved_model1, model2_collision_object) is True:
                    have_contact = True
                    break

        return have_contact

    def have_raytracing_intersections(self, point1, point2, model2):
        """Detects whether there are intersections between the ray(point1, point2) and model2.

        :param point1: The starting point of the ray.
        :param point2: The end point of the ray.
        :param model2: The model2.
        :return: Whether there are intersections between the ray(point1, point2) and model2.
        """
        intersections = model2.ray_trace(point1, point2)[1]
        return intersections.shape[0] > 0

    def classify_blocking_relationship(self, model1, model2):
        """Classifies whether model1 and model2 have blocking relationship.
        This is performed using raytracing.
        The current version only supports rays that are in coordinate orientations, which could be improved
        based on real applications.

        :param model1: The model1.
        :param model2: The model2.
        :return: Whether model1 and model2 have blocking relationship.
        """
        have_blocking = False

        # Iterate each vertex of model1
        for i in range(0, model1.points.shape[0]):
            # The starting point of the ray
            point1 = model1.points[i, :]

            p_x, p_y, p_z = point1[0], point1[1], point1[2]

            # The end points of different rays (the starting point of these rays is the same, which is point1)
            # The orientations of the rays are the coordinate orientations
            point2 = np.array(
                [
                    [self.limit_distance, p_y, p_z],
                    [-self.limit_distance, p_y, p_z],
                    [p_x, self.limit_distance, p_z],
                    [p_x, -self.limit_distance, p_z],
                    [p_x, p_y, self.limit_distance],
                    [p_x, p_y, -self.limit_distance],
                ]
            )

            # If at least one ray intersects with another model, the classification result is True
            for j in range(0, 6):
                if (
                    self.have_raytracing_intersections(point1, point2[j, :], model2)
                    is True
                ):
                    have_blocking = True
                    break

            if have_blocking is True:
                break

        return have_blocking


if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

    relationship_classifier = RelationshipClassifier()

    model1 = pv.Cylinder((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 2.5, 5.0).triangulate()
    model2 = pv.Cylinder((5.1, 0.0, 0.0), (1.0, 0.0, 0.0), 2.5, 5.0).triangulate()

    # print(relationship_classifier.classify_contact_relationship(model1, model2))

    print(relationship_classifier.classify_blocking_relationship(model1, model2))
