from enum import Enum

import numpy as np
import ot
import torch
from matplotlib.path import Path
from scipy import stats
from scipy.spatial.distance import euclidean
import cv2

# From https://gist.github.com/larsmans/3116927
_SQRT2 = np.sqrt(2)  # sqrt(2) with default precision np.float64


def hellinger(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2


def z_score(data_point, dist_mean, dist_std):
    z_score = (data_point - dist_mean) / np.maximum(dist_std, 1e-6)

    # Calculate the p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    return z_score, p_value


def calculate_displacement(
    gt_goal, gt_future, future_samples, waypoint_samples, resize
):
    fde = (
        (
            (((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5
        ).min(dim=0)[0]
        # .mean()
        # .item()
    )
    ade = (
        ((((gt_future - future_samples) / resize) ** 2).sum(dim=3) ** 0.5)
        .mean(dim=2)
        .min(dim=0)[0]
        # .mean()
        # .item()
    )
    return ade.numpy().reshape((-1,)), fde.numpy().reshape((-1,))


def calculate_displacement_mean(
    gt_goal, gt_future, future_samples, waypoint_samples, resize
):
    fde = (
        ((((gt_goal - waypoint_samples[:, :, -1:]) / resize) ** 2).sum(dim=3) ** 0.5)
        # .min(dim=0)[0]
        .mean(dim=0)
        # .mean()
        # .item()
    )
    ade = (
        ((((gt_future - future_samples) / resize) ** 2).sum(dim=3) ** 0.5)
        .mean(dim=2)
        # .min(dim=0)[0]
        .mean(dim=0)
        # .mean()
        # .item()
    )
    return ade.numpy().reshape((-1,)), fde.numpy().reshape((-1,))


# def flip_traj_x(traj, max):
#     traj = traj.detach().clone()
#     traj[:, :, 0] = max - traj[:, :, 0]
#     return traj


# def flip_traj_y(traj, max):
#     traj = traj.detach().clone()
#     traj[:, :, 1] = max - traj[:, :, 1]
#     return traj


def flip_in_bounds(arr, x, y, axis):
    arr_copy = arr.detach().clone()

    content = arr_copy[:, :, :x, :y]

    # Actual manipulation

    # Flip U/D
    arr_copy[:, :, :x, :y] = torch.flip(content, (axis,))
    return arr_copy


def calculate_distances(trajectories1, trajectories2):
    """Calculation of Wasserstein distance + avg. min. distance between trajectories."""
    # Flatten trajectories for computing pairwise distance
    flat_trajectories1 = trajectories1.reshape(trajectories1.shape[0], -1)
    flat_trajectories2 = trajectories2.reshape(trajectories2.shape[0], -1)

    # Define the cost matrix (Euclidean distance between trajectories)
    pairwise_distances = ot.dist(
        flat_trajectories1, flat_trajectories2, metric="euclidean"
    )

    # Normalize the cost matrix to [0, 1]
    cost_matrix = pairwise_distances / pairwise_distances.max()

    # Uniform distributions over the trajectories
    a = np.ones((trajectories1.shape[0],)) / trajectories1.shape[0]
    b = np.ones((trajectories2.shape[0],)) / trajectories2.shape[0]

    # Compute the Wasserstein distance with entropic regularization (Sinkhorn)
    lambda_reg = 5  # Regularization parameter
    transport_plan = ot.bregman.sinkhorn(a, b, cost_matrix, lambda_reg, numItermax=2500)

    # Compute the Wasserstein distance
    wasserstein_distance = np.sum(transport_plan * pairwise_distances)

    # Average minimal distance to the next trajectory
    # Approximation of OT, but OT is a matching.
    # Here we only take the nearest neighbor
    average_min_distance = pairwise_distances.min(axis=-1).mean()

    return wasserstein_distance, average_min_distance


def make_copy(d):
    if isinstance(d, torch.Tensor):
        return d.detach().clone()
    else:
        return d.copy()


def flip_ud_reproj(
    input_trajectory_f,
    pred_traj_f,
    pred_waypoint_map_sigmoid_f,
    waypoint_samples_f,
    future_samples_f,
    size_unpadded,
):
    # Reproject into original image
    input_trajectory_f_re = make_copy(input_trajectory_f)
    input_trajectory_f_re[..., 1] = size_unpadded[0] - input_trajectory_f[..., 1]

    pred_traj_f_re = make_copy(pred_traj_f)
    pred_traj_f_re[..., 1] = size_unpadded[0] - pred_traj_f[..., 1]

    pred_waypoint_map_sigmoid_f_re = np.flip(pred_waypoint_map_sigmoid_f, 2)

    waypoint_samples_f_re = make_copy(waypoint_samples_f)
    waypoint_samples_f_re[:, :, -1:, 1] = (
        size_unpadded[0] - waypoint_samples_f_re[:, :, -1:, 1]
    )

    future_samples_f_re = make_copy(future_samples_f)
    future_samples_f_re[:, :, :, 1] = size_unpadded[0] - future_samples_f_re[:, :, :, 1]

    return (
        input_trajectory_f_re,
        pred_traj_f_re,
        pred_waypoint_map_sigmoid_f_re,
        waypoint_samples_f_re,
        future_samples_f_re,
    )


def flip_lr_reproj(
    input_trajectory_f,
    pred_traj_f,
    pred_waypoint_map_sigmoid_f,
    waypoint_samples_f,
    future_samples_f,
    size_unpadded,
):
    # Reproject into original image
    input_trajectory_f_re = make_copy(input_trajectory_f)
    input_trajectory_f_re[..., 0] = size_unpadded[1] - input_trajectory_f_re[..., 0]

    pred_traj_f_re = make_copy(pred_traj_f)
    pred_traj_f_re[..., 0] = size_unpadded[1] - pred_traj_f_re[..., 0]

    pred_waypoint_map_sigmoid_f_re = np.flip(pred_waypoint_map_sigmoid_f, 3)

    waypoint_samples_f_re = make_copy(waypoint_samples_f)
    waypoint_samples_f_re[:, :, -1:, 0] = (
        size_unpadded[1] - waypoint_samples_f_re[:, :, -1:, 0]
    )

    future_samples_f_re = make_copy(future_samples_f)
    future_samples_f_re[:, :, :, 0] = size_unpadded[1] - future_samples_f_re[:, :, :, 0]

    return (
        input_trajectory_f_re,
        pred_traj_f_re,
        pred_waypoint_map_sigmoid_f_re,
        waypoint_samples_f_re,
        future_samples_f_re,
    )


def resize_reproj(
    input_trajectory_f,
    pred_traj_f,
    pred_waypoint_map_sigmoid_f,
    waypoint_samples_f,
    future_samples_f,
    resize_param_src,
    resize_param_f,
    original_size,
):
    input_trajectory_f_re = input_trajectory_f * resize_param_src / resize_param_f

    pred_traj_f_re = pred_traj_f * resize_param_src / resize_param_f

    # TODO This should actually be rescaled, but the value is currently only used for plot
    # factor = resize_param_src / resize_param_f

    new_pwms = []
    for i in range(pred_waypoint_map_sigmoid_f.shape[0]):
        new_pwms.append([])
        for j in range(pred_waypoint_map_sigmoid_f.shape[1]):
            new_pwms[-1].append(
                cv2.resize(
                    pred_waypoint_map_sigmoid_f[i, j],
                    original_size[-2:],
                    # (0, 0),
                    # fx=factor,
                    # fy=factor,
                    interpolation=cv2.INTER_AREA,
                )
            )

    pred_waypoint_map_sigmoid_f_re = np.stack(new_pwms)

    waypoint_samples_f_re = waypoint_samples_f * resize_param_src / resize_param_f

    future_samples_f_re = future_samples_f * resize_param_src / resize_param_f

    return (
        input_trajectory_f_re,
        pred_traj_f_re,
        pred_waypoint_map_sigmoid_f_re,
        waypoint_samples_f_re,
        future_samples_f_re,
    )


def rotate_reproj(
    input_trajectory_f,
    pred_traj_f,
    pred_waypoint_map_sigmoid_f,
    waypoint_samples_f,
    future_samples_f,
    size_unpadded,
    rotation,
):
    if rotation in {90, 270}:
        center = torch.tensor(size_unpadded[:2][::-1]) / 2
    else:
        center = torch.tensor(size_unpadded[:2]) / 2

    # Reproject into original image
    input_trajectory_f_re = rotate_trajectory(
        torch.from_numpy(input_trajectory_f), image_center=center, degrees=-rotation
    ).numpy()

    pred_traj_f_re = rotate_trajectory(
        torch.from_numpy(pred_traj_f), image_center=center, degrees=-rotation
    ).numpy()

    pred_waypoint_map_sigmoid_f_re = rotate_image(
        torch.from_numpy(pred_waypoint_map_sigmoid_f),
        degrees=-rotation,
        size_unpadded=size_unpadded,
    ).numpy()

    waypoint_samples_f_re = rotate_trajectory(
        torch.from_numpy(waypoint_samples_f), image_center=center, degrees=-rotation
    ).numpy()

    future_samples_f_re = rotate_trajectory(
        torch.from_numpy(future_samples_f), image_center=center, degrees=-rotation
    ).numpy()

    return (
        input_trajectory_f_re,
        pred_traj_f_re,
        pred_waypoint_map_sigmoid_f_re,
        waypoint_samples_f_re,
        future_samples_f_re,
    )


def rotate_image_and_trajectory(image, trajectory, degrees=90, size_unpadded=None):
    """
    Rotates image tensor (b,c,h,w) and trajectory (t,l,2) counter-clockwise by k * 90 degrees around the image center

    Args:
        image: torch.Tensor of shape (b,c,h,w) - batch of images
        trajectory: torch.Tensor of shape (t,l,2) - trajectory coordinates
        degrees: int - number of degrees to rotate

    Returns:
        rotated_image: torch.Tensor - rotated image
        rotated_traj: torch.Tensor - rotated trajectory
    """
    # Get image dimensions
    if size_unpadded is None:
        size_unpadded = image.shape[-2:]

    center = (
        torch.tensor(size_unpadded, dtype=trajectory.dtype, device=trajectory.device)
        / 2
    )
    return rotate_image(image, degrees, size_unpadded), rotate_trajectory(
        trajectory, center, degrees
    )


def rotate_trajectory(trajectory, image_center, degrees=90):
    if degrees < 0:
        degrees += 360

    k = degrees // 90
    assert k in [0, 1, 2, 3], f"Invalid rotation: {degrees} degrees"

    if k == 0:
        return trajectory.clone()

    # Create rotation matrix
    theta = -k * np.pi / 2  # Negative for counter-clockwise rotation
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    R = torch.tensor(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]],
        dtype=trajectory.dtype,
        device=trajectory.device,
    )

    # Center coordinates at image center, rotate, then translate back
    centered_traj = trajectory - torch.flip(image_center, dims=(0,))  # image_center

    # Reshape to (t*l, 2) for batch rotation
    flat_traj = centered_traj.reshape(-1, 2)
    rotated_flat = torch.mm(flat_traj, R.t())
    rotated_traj = rotated_flat.reshape(*trajectory.shape)

    # Translate back based on rotated image dimensions
    if k % 2 == 1:  # 90° or 270° - dimensions are swapped
        rotated_traj = rotated_traj + image_center
    else:  # 180° - dimensions stay the same
        rotated_traj = rotated_traj + torch.flip(image_center, dims=(0,))

    return rotated_traj


def rotate_image(image, degrees=90, size_unpadded=None):
    if degrees < 0:
        degrees += 360

    k = degrees // 90
    assert k in [0, 1, 2, 3], f"Invalid rotation: {degrees} degrees"

    if k == 0:
        return image.clone()

    if size_unpadded is None:
        size_unpadded = image.shape[-2:]

    rotated_image = torch.zeros_like(image)
    content = image[..., : size_unpadded[0], : size_unpadded[1]]

    if k == 2:
        new_h, new_w = size_unpadded[0], size_unpadded[1]
    else:
        new_h, new_w = size_unpadded[1], size_unpadded[0]

    rotated_image = torch.rot90(rotated_image, k, dims=[-2, -1])
    rotated_image[..., :new_h, :new_w] = torch.rot90(content, k, dims=[-2, -1])

    return rotated_image


class ClassChangeType(Enum):
    LESS_WALKABLE = "less_walkable"
    MORE_WALKABLE = "more_walkable"
    OBSTACLE = "obstacle"


segmentation_class_map = {
    "background": 0,
    "pavement": 1,
    "road": 2,
    "structure": 3,
    "terrain": 4,
    "tree": 5,
}


def random_class_change(scene_image_segm, change_type=None):
    class_changes = [
        ("pavement", "road", ClassChangeType.LESS_WALKABLE),
        ("terrain", "road", ClassChangeType.LESS_WALKABLE),
        ("road", "pavement", ClassChangeType.MORE_WALKABLE),
        ("road", "terrain", ClassChangeType.MORE_WALKABLE),
        ("structure", "road", ClassChangeType.MORE_WALKABLE),
        ("structure", "pavement", ClassChangeType.MORE_WALKABLE),
        ("structure", "terrain", ClassChangeType.MORE_WALKABLE),
        ("tree", "road", ClassChangeType.MORE_WALKABLE),
        ("tree", "pavement", ClassChangeType.MORE_WALKABLE),
        ("tree", "terrain", ClassChangeType.MORE_WALKABLE),
        ("road", "structure", ClassChangeType.OBSTACLE),
        ("road", "tree", ClassChangeType.OBSTACLE),
        ("pavement", "structure", ClassChangeType.OBSTACLE),
        ("pavement", "tree", ClassChangeType.OBSTACLE),
        ("terrain", "structure", ClassChangeType.OBSTACLE),
        ("terrain", "tree", ClassChangeType.OBSTACLE),
    ]

    # The segmentation is in float, we extract the most likely class
    class_labels = scene_image_segm.argmax(dim=1)

    # Only take changes for classes that occur in the segmentation
    applicable = []

    for cc in (cc for cc in class_changes if (cc[2] == change_type) or change_type is None):
        if (class_labels == segmentation_class_map[cc[0]]).any():
            applicable.append(cc)

    # Randomly select an applicable change from the list
    class_change_idx = np.random.choice(len(applicable))
    src_class, tgt_class, change_type = applicable[class_change_idx]

    src_idx = segmentation_class_map[src_class]
    tgt_idx = segmentation_class_map[tgt_class]
    class_condition = (class_labels == src_idx)[0]

    # Modify actual segmentation values
    scene_image_segm_modified = scene_image_segm.detach().clone()
    tmp_var = scene_image_segm_modified[0, tgt_idx, class_condition]

    scene_image_segm_modified[0, src_idx, class_condition] = scene_image_segm_modified[
        0, tgt_idx, class_condition
    ]
    scene_image_segm_modified[0, tgt_idx, class_condition] = tmp_var
    affected_area = int(class_condition.sum())

    # print(
    #     f"Changing {src_class} ({src_idx}) to {tgt_class} ({tgt_idx}) - {change_type.value} ({affected_area / scene_image_segm_modified.numel() * 100:.3f}% of pixels)"
    # )

    return scene_image_segm_modified, {
        "src_class": src_class,
        "tgt_class": tgt_class,
        "change_type": change_type,
        "affected_area": affected_area,
        "class_condition": class_condition.cpu().numpy(),
    }


def create_obstacle_mask(
    size, obstacle_min_y, obstacle_max_y, obstacle_min_x, obstacle_max_x
):
    """Create a binary mask for an circle-approximation of given size."""
    mask = np.zeros((size, size), dtype=bool)
    center = size // 2
    radius = size // 2

    # Create points for the corners
    angles = np.linspace(0, 2 * np.pi, 13)[
        :-1
    ]  # n points, excluding the last to close the shape
    points = np.array(
        [
            (center + radius * np.cos(angle), center + radius * np.sin(angle))
            for angle in angles
        ]
    ).astype(int)

    # Fill the obstacle
    path = Path(points)
    y, x = np.mgrid[0:size, 0:size]
    points = np.column_stack((x.ravel(), y.ravel()))
    mask = path.contains_points(points).reshape(size, size)

    # Calculate the actual region size
    region_height = obstacle_max_y - obstacle_min_y
    region_width = obstacle_max_x - obstacle_min_x

    # Resize the mask to match the actual region size
    if region_height != size or region_width != size:
        # Crop the mask to match the actual region size
        mask_start_y = (size - region_height) // 2 if region_height < size else 0
        mask_start_x = (size - region_width) // 2 if region_width < size else 0
        mask = mask[
            mask_start_y : mask_start_y + region_height,
            mask_start_x : mask_start_x + region_width,
        ]

    return mask


def add_obstacle(
    scene_image_segm,
    trajectory,
    distance: float = 0.5,
    obstacle_size: int = 20,
    obstacle_class: str = "structure",
):
    """
    Add an obstacle to the class labels at a specified distance in the (future) trajectory.

    Args:
        class_labels: numpy array of class labels
        trajectory: numpy array of trajectory points
    """
    assert obstacle_class in segmentation_class_map, (
        f"Invalid obstacle class: {obstacle_class}"
    )

    scene_image_segm_modified = scene_image_segm.detach().clone()

    num_points = len(trajectory)

    # Calculate the boundaries
    obstacle_center = trajectory[int(distance * num_points)]
    obstacle_min_x = max(int(obstacle_center[0] - obstacle_size // 2), 0)
    obstacle_max_x = min(
        int(obstacle_center[0] + obstacle_size // 2),
        scene_image_segm_modified.shape[-1],
    )
    obstacle_min_y = max(int(obstacle_center[1] - obstacle_size // 2), 0)
    obstacle_max_y = min(
        int(obstacle_center[1] + obstacle_size // 2),
        scene_image_segm_modified.shape[-2],
    )

    # Apply the mask
    obstacle_mask = create_obstacle_mask(
        obstacle_size, obstacle_min_y, obstacle_max_y, obstacle_min_x, obstacle_max_x
    )
    obstacle_mask = torch.from_numpy(obstacle_mask).to(scene_image_segm_modified.device)

    class_idx = segmentation_class_map[obstacle_class]

    scene_image_segm_modified[
        0, class_idx, obstacle_min_y:obstacle_max_y, obstacle_min_x:obstacle_max_x
    ][obstacle_mask] = 1

    # Normalize the segmentation values again to sum to 1
    scene_image_segm_modified[
        0, :, obstacle_min_y:obstacle_max_y, obstacle_min_x:obstacle_max_x
    ] = scene_image_segm_modified[
        0, :, obstacle_min_y:obstacle_max_y, obstacle_min_x:obstacle_max_x
    ] / scene_image_segm_modified[
        0, :, obstacle_min_y:obstacle_max_y, obstacle_min_x:obstacle_max_x
    ].sum(dim=0, keepdim=True)

    full_mask = torch.zeros(scene_image_segm_modified.shape[-2:], dtype=bool)
    full_mask[obstacle_min_y:obstacle_max_y, obstacle_min_x:obstacle_max_x] = (
        obstacle_mask
    )
    return scene_image_segm_modified, {
        "obstacle_distance": distance,
        "obstacle_size": obstacle_size,
        "obstacle_class": obstacle_class,
        "obstacle_mask": full_mask.cpu().numpy(),
    }


def check_intersection(traj, full_obstacle_mask):
    return full_obstacle_mask[traj[:, 1].astype(int), traj[:, 0].astype(int)].any()
