# %%
%load_ext autoreload
%autoreload 2

import pandas as pd
import yaml
import torch
from model import YNet
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from utils.dataloader import SceneDataset, scene_collate
from utils.preprocessing import create_images_dict
from utils.image_utils import (
    resize,
    pad,
    preprocess_image_for_segmentation,
    create_dist_mat,
)
from predict import predict
from mt import (
    flip_in_bounds,
    calculate_distances,
    calculate_displacement,
    calculate_displacement_mean,
    z_score,
    flip_ud_reproj,
    flip_lr_reproj,
    resize_reproj,
    rotate_image_and_trajectory,
    rotate_reproj,
    random_class_change,
    segmentation_class_map,
    add_obstacle,
)
import scipy.special

import itertools
import json

# %% [markdown]
# #### Some hyperparameters and settings

# %%
## Longterm Prediction Setting
# CONFIG_FILE_PATH = 'config/sdd_longterm.yaml'  # yaml config file containing all the hyperparameters
# DATASET_NAME = 'sdd'

# TEST_DATA_PATH = 'data/SDD/test_longterm.pkl'
# TEST_IMAGE_PATH = 'data/SDD/test'
# OBS_LEN = 5  # in timesteps
# PRED_LEN = 30  # in timesteps
# NUM_GOALS = 20  # K_e
# NUM_TRAJ = 5  # K_a

## Shortterm Prediction Setting
CONFIG_FILE_PATH = (
    "config/sdd_trajnet.yaml"  # yaml config file containing all the hyperparameters
)
DATASET_NAME = "sdd"

TEST_DATA_PATH = "data/SDD/test_trajnet.pkl"
TEST_IMAGE_PATH = "data/SDD/test"  # only needed for YNet, PECNet ignores this value
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

## General Settings
ROUNDS = 3  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 8

## Testing Settings
ROUNDS_VARIATION = 8  # N
RESIZE_PARAMS = (0.2, 0.3)
ROTATION_PARAMS = (90, 180, 270)

# %% [markdown]
# #### Load config file and print hyperparameters

# %%
with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split(".yaml")[0].split("config/")[1]
params

# %% [markdown]
# #### Load preprocessed Data

# %%
df_test = pd.read_pickle(TEST_DATA_PATH)

# %%
df_test.head()

# %%
df_test.shape

# %% [markdown]
# #### Initiate model and load pretrained weights

# %%
ynet_model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)

# %%
ynet_model.load(f"pretrained_models/{experiment_name}_weights.pt")

# %%
# Preparation code from `model.evaluate`
device = None
dataset_name = DATASET_NAME
image_path = TEST_IMAGE_PATH
batch_size = BATCH_SIZE
rounds = (ROUNDS,)
num_goals = NUM_GOALS
num_traj = NUM_TRAJ

if device is None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_len = ynet_model.obs_len
pred_len = ynet_model.pred_len
total_len = pred_len + obs_len

print("Preprocess data")
dataset_name = dataset_name.lower()
if dataset_name == "sdd":
    image_file_name = "reference.jpg"
elif dataset_name == "ind":
    image_file_name = "reference.png"
elif dataset_name == "eth":
    image_file_name = "oracle.png"
else:
    raise ValueError(f"{dataset_name} dataset is not supported")

# ETH/UCY specific: Homography matrix is needed to convert pixel to world coordinates
if dataset_name == "eth":
    ynet_model.homo_mat = {}
    for scene in [
        "eth",
        "hotel",
        "students001",
        "students003",
        "uni_examples",
        "zara1",
        "zara2",
        "zara3",
    ]:
        ynet_model.homo_mat[scene] = torch.Tensor(
            np.loadtxt(f"data/eth_ucy/{scene}_H.txt")
        ).to(device)
    seg_mask = True
else:
    ynet_model.homo_mat = None
    seg_mask = False

test_images = create_images_dict(
    df_test, image_path=image_path, image_file=image_file_name
)
test_images_orig = create_images_dict(
    df_test, image_path=image_path, image_file=image_file_name
)
resize(test_images_orig, factor=params["resize"], seg_mask=seg_mask)

test_dataset = SceneDataset(df_test, resize=params["resize"], total_len=total_len)
test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=scene_collate)

# Preprocess images, in particular resize, pad and normalize as semantic segmentation backbone requires
resize(test_images, factor=params["resize"], seg_mask=seg_mask)

# Size before padding, needed to extract segmentation map for manipulation
size_unpadded = {k: v.shape for k, v in test_images.items()}

pad(
    test_images, division_factor=ynet_model.division_factor
)  # make sure that image shape is divisible by 32, for UNet architecture
preprocess_image_for_segmentation(test_images, seg_mask=seg_mask)

model = ynet_model.model.to(device)

# Create template
size = int(4200 * params["resize"])

input_template = torch.Tensor(create_dist_mat(size=size)).to(device)


# Rescale preparation
rescale_mr = {}

for resize_param in RESIZE_PARAMS:
    test_images_r = create_images_dict(
        df_test, image_path=image_path, image_file=image_file_name
    )
    resize(test_images_r, factor=resize_param, seg_mask=seg_mask)

    size_unpadded_resized = {k: v.shape for k, v in test_images_r.items()}

    pad(test_images_r, division_factor=ynet_model.division_factor)
    preprocess_image_for_segmentation(test_images_r, seg_mask=seg_mask)
    rescale_mr[resize_param] = (test_images_r, size_unpadded_resized)

# %%
for trajectory, meta, scene in test_loader:
    print(scene, len(trajectory))

# %% [markdown]
# #### Evaluate model

# %%
model.eval()

scene_name = "little_1"


def get_scene(loader, scene_name):
    for trajectory, meta, scene in loader:
        if scene == scene_name:
            return trajectory, meta, scene


log = []

with torch.no_grad():
    trajectory, meta, scene = get_scene(test_loader, scene_name)

    # Get scene image and apply semantic segmentation
    scene_image = test_images[scene].to(device).unsqueeze(0)
    scene_image_segm = model.segmentation(scene_image)

    scene_image_segm_resized = {}
    for resize_param_f in RESIZE_PARAMS:
        (test_images_r, size_unpadded_resized) = rescale_mr[resize_param_f]
        scene_image_segm_resized[resize_param_f] = model.segmentation(
            test_images_r[scene].to(device).unsqueeze(0)
        )

    for input_traj_id in range(0, len(trajectory), batch_size):  # len(trajectory)
        gt_future = trajectory[input_traj_id : input_traj_id + batch_size, obs_len:]
        input_trajectory = trajectory[
            input_traj_id : input_traj_id + batch_size, :obs_len, :
        ]
        gt_goal = gt_future[:, -1:]

        source_results = []

        # TODO Instead of this variation we could measure the variation in the 20 trajs per goals (5)
        for _ in range(ROUNDS_VARIATION):
            # We measure multiple times to get an idea of the intra-model variation in the results
            # This will become the threshold for the MR evaluation
            r = predict(
                model=model,
                scene_image=scene_image_segm,
                input_trajectory=input_trajectory,
                waypoints=params["waypoints"],
                num_goals=num_goals,
                num_traj=num_traj,
                input_template=input_template,
                obs_len=obs_len,
                temperature=params["temperature"],
                rel_thresh=0.002,
                device=device,
                use_CWS=True if len(params["waypoints"]) > 1 else False,
                use_TTST=True,
                CWS_params=params["CWS_params"],
            )
            r = [s.cpu().numpy() for s in r]
            ade, fde = calculate_displacement(
                gt_goal,
                gt_future,
                r[1],
                r[2],
                resize=params["resize"],
            )
            ade_mean, fde_mean = calculate_displacement_mean(
                gt_goal,
                gt_future,
                r[1],
                r[2],
                resize=params["resize"],
            )

            source_results.append(r + [ade, fde, ade_mean, fde_mean])

        # Calculate the avg. Wasserstein distance between the predictions per input trajectory
        source_distances = np.zeros(
            (int(scipy.special.comb(ROUNDS_VARIATION, 2)), input_trajectory.shape[0])
        )

        for pix, (r1, r2) in enumerate(itertools.combinations(source_results, 2)):
            future_samples1 = r1[1]
            future_samples2 = r2[1]

            for trajix in range(input_trajectory.shape[0]):
                source_distances[pix, trajix] = calculate_distances(
                    future_samples1[:, trajix, ...], future_samples2[:, trajix, ...]
                )[0]

        # (batch_size,)
        src_dist_mean = np.mean(source_distances, axis=0)
        src_dist_std = np.std(source_distances, axis=0)

        # Pick source test case
        # Source test case is the input_trajectory + scene_image_segm from above

        # Create follow-up input
        scene_image_segm_f = scene_image_segm.detach().clone()
        input_trajectory_f = input_trajectory.detach().clone()

        # Actual manipulation

        # Flip U/D
        # scene_image_segm_f = flip_in_bounds(
        #     scene_image_segm,
        #     x=size_unpadded[scene][0],
        #     y=size_unpadded[scene][1],
        #     axis=2,
        # )
        # input_trajectory_f[:, :, 1] = (
        #     size_unpadded[scene][0] - input_trajectory_f[:, :, 1]
        # )

        # Flip L/R
        # scene_image_segm_f = flip_in_bounds(
        #     scene_image_segm,
        #     x=size_unpadded[scene][0],
        #     y=size_unpadded[scene][1],
        #     axis=3,
        # )
        # input_trajectory_f[:, :, 0] = (
        #     size_unpadded[scene][1] - input_trajectory_f[:, :, 0]
        # )

        # TODO Rotate -> do before segmentation
        # Params: [90, 180, 270]
        # rotation_param = ROTATION_PARAMS[1]
        # center_x = size_unpadded[scene][0] / 2
        # center_y = size_unpadded[scene][1] / 2
        # input_trajectory_f = rotate_around_point_torch(
        #     input_trajectory_f, degrees=rotation_param, origin=(center_x, center_y)
        # )
        # scene_image_segm_f = rotate_tensor(
        #     scene_image_segm, rotation_param, size_unpadded[scene]
        # )

        # Resize
        # resize_param_f = RESIZE_PARAMS[1]
        # input_trajectory_f = input_trajectory_f * resize_param_f / params["resize"]
        # scene_image_segm_f = scene_image_segm_resized[resize_param_f].detach().clone()

        # Revert: input_trajectory_f = input_trajectory_f * params["resize"] / resize_value
        # If instable, create dataset and reload original trajectories for resizing

        # Segmentation class change
        # scene_image_segm_f, cc_info = random_class_change(scene_image_segm)

        # Obstacle appearance
        obstacle_distance = 0.4
        scene_image_segm_f, parameter_info = add_obstacle(scene_image_segm, input_trajectory[0], obstacle_distance
        )
        # input_trajectory_f_re = input_trajectory_f.clone()

        # TODO Keep track of change expectation / output relation
        # Eq, Red, Inc, Avoid?

        (
            pred_traj_f,
            future_samples_f,
            waypoint_samples_f,
            pred_waypoint_map_sigmoid_f,
        ) = predict(
            model=model,
            scene_image=scene_image_segm_f,
            input_trajectory=input_trajectory_f,
            waypoints=params["waypoints"],
            num_goals=num_goals,
            num_traj=num_traj,
            input_template=input_template,
            obs_len=obs_len,
            temperature=params["temperature"],
            rel_thresh=0.002,
            device=device,
            use_CWS=True if len(params["waypoints"]) > 1 else False,
            use_TTST=True,
            CWS_params=params["CWS_params"],
        )

        input_trajectory = input_trajectory.cpu().numpy()
        input_trajectory_f = input_trajectory_f.cpu().numpy()
        scene_image_segm_f = scene_image_segm_f.cpu().numpy()
        pred_traj_f = pred_traj_f.cpu().numpy()
        future_samples_f = future_samples_f.cpu().numpy()
        waypoint_samples_f = waypoint_samples_f.cpu().numpy()
        pred_waypoint_map_sigmoid_f = pred_waypoint_map_sigmoid_f.cpu().numpy()

        input_trajectory_f_re = input_trajectory_f #.clone()
        pred_traj_f_re = pred_traj_f #.clone()
        pred_waypoint_map_sigmoid_f_re= pred_waypoint_map_sigmoid_f #.clone()
        waypoint_samples_f_re = waypoint_samples_f #.clone()
        future_samples_f_re =   future_samples_f #.clone()

        ## Evaluation
        # Revert metamorphic transformation
        # (
        #     input_trajectory_f_re,
        #     pred_traj_f_re,
        #     pred_waypoint_map_sigmoid_f_re,
        #     waypoint_samples_f_re,
        #     future_samples_f_re,
        # ) = flip_ud_reproj(
        #     input_trajectory_f,
        #     pred_traj_f=pred_traj_f,
        #     pred_waypoint_map_sigmoid_f=pred_waypoint_map_sigmoid_f,
        #     waypoint_samples_f=waypoint_samples_f,
        #     future_samples_f=future_samples_f,
        #     size_unpadded=size_unpadded[scene],
        # )

        # (
        #     input_trajectory_f_re,
        #     pred_traj_f_re,
        #     pred_waypoint_map_sigmoid_f_re,
        #     waypoint_samples_f_re,
        #     future_samples_f_re,
        # ) = flip_lr_reproj(
        #     input_trajectory_f,
        #     pred_traj_f=pred_traj_f,
        #     pred_waypoint_map_sigmoid_f=pred_waypoint_map_sigmoid_f,
        #     waypoint_samples_f=waypoint_samples_f,
        #     future_samples_f=future_samples_f,
        #     size_unpadded=size_unpadded[scene],
        # )

        # (
        #     input_trajectory_f_re,
        #     pred_traj_f_re,
        #     pred_waypoint_map_sigmoid_f_re,
        #     waypoint_samples_f_re,
        #     future_samples_f_re,
        # ) = resize_reproj(
        #     input_trajectory_f,
        #     pred_traj_f=pred_traj_f,
        #     pred_waypoint_map_sigmoid_f=pred_waypoint_map_sigmoid_f,
        #     waypoint_samples_f=waypoint_samples_f,
        #     future_samples_f=future_samples_f,
        #     resize_param_src=params["resize"],
        #     resize_param_f=resize_param_f,
        # )

        # (
        #     input_trajectory_f_re,
        #     pred_traj_f_re,
        #     pred_waypoint_map_sigmoid_f_re,
        #     waypoint_samples_f_re,
        #     future_samples_f_re,
        # ) = rotate_reproj(
        #     input_trajectory_f,
        #     pred_traj_f=pred_traj_f,
        #     pred_waypoint_map_sigmoid_f=pred_waypoint_map_sigmoid_f,
        #     waypoint_samples_f=waypoint_samples_f,
        #     future_samples_f=future_samples_f,
        #     size_unpadded=size_unpadded[scene],
        #     rotation=rotation_param,
        # )

        assert np.isclose(input_trajectory_f_re, input_trajectory).all(), (
            "Reprojected input doesn't match original input"
        )

        # Here, we can evaluate against all samples collected before

        # For reference, evaluate against ground-truth labels, because we have them
        # Not part of the actual MR which doesn't need labels
        ade_f, fde_f = calculate_displacement(
            gt_goal,
            gt_future,
            future_samples_f_re,
            waypoint_samples_f_re,
            resize=params["resize"],
        )
        ade_f_mean, fde_f_mean = calculate_displacement_mean(
            gt_goal,
            gt_future,
            future_samples_f_re,
            waypoint_samples_f_re,
            resize=params["resize"],
        )

        # TODO We can do follow-ups per source test case and still use the same acceptance criteria

        for input_idx in range(input_trajectory.shape[0]):
            for src_ix, src_result in enumerate(source_results):
                (
                    pred_traj,
                    future_samples,
                    waypoint_samples,
                    pred_waypoint_map_sigmoid,
                    ade,
                    fde,
                    ade_mean,
                    fde_mean,
                ) = src_result

                ade_wasserstein, ade_min_distance = calculate_distances(
                    trajectories1=future_samples_f_re[:, input_idx, ...],
                    trajectories2=future_samples[:, input_idx, ...],
                )
                fde_wasserstein, fde_min_distance = calculate_distances(
                    waypoint_samples_f_re[:, input_idx, -1, :],
                    waypoint_samples[:, input_idx, -1, :],
                )
                zsc, pval = z_score(
                    ade_wasserstein, src_dist_mean[input_idx], src_dist_std[input_idx]
                )

                log_entry = {
                    "mr": "flipud",
                    "scene": scene,
                    "input_traj_id": input_traj_id + input_idx,
                    "source_idx": src_ix,
                    "follow_idx": 0,
                    "num_sources": len(source_results),
                    "source_dist_mean": src_dist_mean[input_idx],
                    "source_dist_std": src_dist_std[input_idx],
                    "source_fde": fde[input_idx],
                    "source_ade": ade[input_idx],
                    "source_fde_mean": fde_mean[input_idx],
                    "source_ade_mean": ade_mean[input_idx],
                    "ade_wasserstein": ade_wasserstein,
                    "ade_min_distance": ade_min_distance,
                    "fde_wasserstein": fde_wasserstein,
                    "fde_min_distance": fde_min_distance,
                    "follow_fde": fde_f[input_idx],
                    "follow_ade": ade_f[input_idx],
                    "follow_fde_mean": fde_f_mean[input_idx],
                    "follow_ade_mean": ade_f_mean[input_idx],
                    "zscore": zsc,
                    "pvalue": pval,
                }
                log.append(log_entry)

            print(
                f"FDE: {fde.mean():.2f} | FDE_f: {fde_f.mean():.2f} | ADE: {ade.mean():.2f} | ADE_f: {ade_f.mean():.2f}"
            )

        break
        # pd.DataFrame.from_records(log).to_csv(f"{experiment_name}_flipud.csv", index=False)

# %%

## Add obstacle demo

src_res_idx = np.random.randint(0, len(source_results))
pred_traj = source_results[src_res_idx][0]
pred_traj_idx = np.random.randint(0, pred_traj.shape[0])
print(pred_traj.shape)

obstacle_distance = np.random.uniform(0.1, 0.9)
seg, fm = add_obstacle(scene_image_segm, pred_traj[pred_traj_idx], obstacle_distance)

fig, ax = plt.subplots(1, 2, figsize=(15, 7.5))

ax[0].imshow(np.transpose(scene_image_segm[0].cpu().numpy(), (1,2,0)).argmax(axis=-1))
ax[0].scatter(pred_traj[pred_traj_idx][:, 0], pred_traj[pred_traj_idx][:, 1], s=2, c='r')

ax[1].imshow(np.transpose(seg[0].cpu().numpy(), (1,2,0)).argmax(axis=-1))
ax[1].scatter(pred_traj[pred_traj_idx][:, 0], pred_traj[pred_traj_idx][:, 1], s=2, c='r')

# %%
(
    pred_waypoint_map_sigmoid_f[:, 0, mask.cpu().numpy()]
    / source_results[0][3][:, 0, mask.cpu().numpy()]
).mean()

# %%
from scipy import stats

stats.wilcoxon(
    pred_waypoint_map_sigmoid_f[:, 0, mask.cpu().numpy()].flatten(),
    source_results[0][3][:, 0, mask.cpu().numpy()].flatten(),
    alternative="greater",
)

# %%
future_samples_f

# %%
import matplotlib.pyplot as plt

plt.imshow(scene_image_segm.argmax(dim=1).squeeze().cpu().numpy())

plt.figure()
plt.imshow(scene_image_segm_f.argmax(dim=1).squeeze().cpu().numpy())

# %%
scene_image_segm.argmax(dim=1)

# %%
pd.DataFrame.from_records(log)

# %%
input_trajectory[0], input_trajectory_f[0]

# %%
cx = size_unpadded[scene][1] / 2
cy = size_unpadded[scene][0] / 2

ipf = rotate_around_point_torch(
    torch.tensor(input_trajectory[0]), degrees=180, origin=(cx, cy)
)
ipf2 = rotate_around_point_torch(ipf, degrees=-180, origin=(cx, cy))
plt.scatter(input_trajectory[0][:, 0], input_trajectory[0][:, 1])
plt.scatter(cx, cy, c="black")
plt.scatter(ipf[:, 0], ipf[:, 1], c="red")
# plt.scatter(ipf2[:, 0], ipf2[:, 1], c="green")

# %%
# For plotting, pick one trajectory
ix = 0
inp = input_trajectory[ix]
pred = pred_traj[ix]
truth = gt_future[ix]
heatmap = pred_waypoint_map_sigmoid[ix, -1]

inp_f = input_trajectory_f_re[ix]
pred_f = pred_traj_f[ix]
# truth_f = gt_future[ix]
heatmap_f = pred_waypoint_map_sigmoid_f_re[ix, -1]

plt.figure(figsize=(8, 12))

# plt.imshow(np.flip(test_images_orig[scene_name], axis=1))
plt.imshow(test_images_orig[scene_name])
# plt.imshow(
#     heatmap,
#     cmap="bwr",
#     interpolation="nearest",
#     alpha=0.2,
# )

# plt.imshow(
#     heatmap_f,
#     cmap="PRGn",
#     interpolation="nearest",
#     alpha=0.2,
# )

scatter_size = 2.5

plt.scatter(inp[:, 0], inp[:, 1], c="blue", s=scatter_size, label="Past Trajectory")

plt.scatter(
    future_samples[:, ix, :, 0],
    future_samples[:, ix, :, 1],
    c="red",
    s=scatter_size,
    label="Predicted Trajectories",
    # marker=">",
    # alpha=0.5,
)
# plt.scatter(
#     future_samples_f_re[:, ix, :, 0],
#     future_samples_f_re[:, ix, :, 1],
#     c="green",
#     s=scatter_size,
#     label="Follow-Up (alt)",
#     # marker="x",
#     alpha=0.5,
# )

# plt.scatter(pred[:, 0], pred[:, 1], c="red", s=scatter_size, label="Source")
plt.scatter(truth[:, 0], truth[:, 1], c="yellow", s=scatter_size, label="Truth")


plt.scatter(pred_f[:, 0], pred_f[:, 1], c="black", s=scatter_size, label="Follow-Up")
# plt.scatter(
#     inp_f[:, 0],
#     inp_f[:, 1],
#     c="purple",
#     s=scatter_size,
# )
# plt.legend()
plt.axis("off")
# plt.scatter(
#     waypoint_samples[:, ix, -1:, 0],
#     waypoint_samples[:, ix, -1:, 1],
#     c="brown",
#     s=5,
# )

# plt.scatter(
#     waypoint_samples_ff[:, ix, -1:, 0],
#     waypoint_samples_ff[:, ix, -1:, 1],
#     c="purple",
#     s=5,
# )
plt.tight_layout()
# plt.savefig("traj-inp-out.pdf", dpi=400, bbox_inches="tight")
fig = plt.gcf()
fig.canvas.draw()
data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

# %%
import cv2

res = cv2.resize(data, dsize=(800, 600))
cv2.imwrite("input-output-example.png", cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

# %%
plt.imshow(test_images_orig[scene_name])

# %%
def rotate_image_and_trajectory2(image, trajectory, degrees=90):
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
    import torch
    import numpy as np

    image = image.clone()
    trajectory = trajectory.clone()

    # Validate degrees
    k = degrees // 90
    k = k % 4  # Normalize to 0-3 range
    if k == 0:
        return image, trajectory

    # Get image dimensions
    h, w = image.shape[-2:]

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
    t, l, _ = trajectory.shape
    image_center = torch.tensor(
        [w / 2, h / 2], dtype=trajectory.dtype, device=trajectory.device
    )
    centered_traj = trajectory - image_center

    # Reshape to (t*l, 2) for batch rotation
    flat_traj = centered_traj.reshape(-1, 2)
    rotated_flat = torch.mm(flat_traj, R.t())
    rotated_traj = rotated_flat.reshape(t, l, 2)

    # Translate back based on rotated image dimensions
    if k % 2 == 1:  # 90° or 270° - dimensions are swapped
        rotated_traj = rotated_traj + torch.tensor(
            [h / 2, w / 2], dtype=trajectory.dtype, device=trajectory.device
        )
    else:  # 180° - dimensions stay the same
        rotated_traj = rotated_traj + image_center

    # Rotate image tensor around its center
    rotated_image = torch.rot90(image, k, dims=[-2, -1])

    return rotated_image, rotated_traj


# %%
from mt import rotate_image_and_trajectory

inp_f = torch.from_numpy(input_trajectory_f)
pred_f = pred_traj_f[ix]
# truth_f = gt_future[ix]
heatmap_f = pred_waypoint_map_sigmoid_f[ix, -1]
print(inp_f.shape)
print(size_unpadded[scene], test_images_orig[scene_name].shape)
inpimg = torch.from_numpy(test_images_orig[scene_name]).permute(2, 0, 1)

rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=0)
rotimg = rotimg.permute(1, 2, 0)
print(inpimg.shape, rotimg.shape, rottraj.shape)
plt.imshow(rotimg)
plt.scatter(
    rottraj[..., 0],
    rottraj[..., 1],
    c="purple",
    s=scatter_size,
)

plt.figure()
rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=90)
rotimg = rotimg.permute(1, 2, 0)
print(inpimg.shape, rotimg.shape, rottraj.shape)
plt.imshow(rotimg)
plt.scatter(
    rottraj[..., 0],
    rottraj[..., 1],
    c="purple",
    s=scatter_size,
)

plt.figure()
rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=180)
rotimg = rotimg.permute(1, 2, 0)
print(inpimg.shape, rotimg.shape, rottraj.shape)
plt.imshow(rotimg)
plt.scatter(
    rottraj[..., 0],
    rottraj[..., 1],
    c="purple",
    s=scatter_size,
)

plt.figure()
rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=270)
rotimg = rotimg.permute(1, 2, 0)
print(inpimg.shape, rotimg.shape, rottraj.shape)
plt.imshow(rotimg)
plt.scatter(
    rottraj[..., 0],
    rottraj[..., 1],
    c="purple",
    s=scatter_size,
)

# %%
## Simple test of rotation functions

inp_f = torch.from_numpy(input_trajectory_f)
pred_f = pred_traj_f[ix]
# truth_f = gt_future[ix]
heatmap_f = pred_waypoint_map_sigmoid_f[ix, -1]

inpimg = torch.from_numpy(test_images_orig[scene_name]).permute(2, 0, 1)

rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=0)

for _ in range(4):
    rotimg, rottraj = rotate_image_and_trajectory(rotimg, rottraj, degrees=90)

assert torch.all(inpimg == rotimg)
assert torch.all(inp_f == rottraj)

rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=0)

for _ in range(2):
    rotimg, rottraj = rotate_image_and_trajectory(rotimg, rottraj, degrees=180)

assert torch.all(inpimg == rotimg)
assert torch.all(inp_f == rottraj)

# 270 == 3*90
rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=0)

for _ in range(3):
    rotimg, rottraj = rotate_image_and_trajectory(rotimg, rottraj, degrees=90)

rotimg2, rottraj2 = rotate_image_and_trajectory(inpimg, inp_f, degrees=270)
assert torch.all(rotimg2 == rotimg)
assert torch.all(rottraj2 == rottraj)

rotimg, rottraj = rotate_image_and_trajectory(inpimg, inp_f, degrees=90)
rotimg2, rottraj2 = rotate_image_and_trajectory(rotimg, rottraj, degrees=-90)
print(inpimg.shape, rotimg2.shape, inp_f.shape, rottraj2.shape)
assert torch.all(rotimg2 == inpimg)
assert torch.all(rottraj2 == inp_f)

# %%
inp_f - rottraj2

# %%
# .shape
# plt.imshow(scene_image_segm)
data = np.transpose(scene_image_segm.cpu().numpy().argmax(axis=1), (1, 2, 0))
class_labels = np.squeeze(data)

# Create a color map with distinct colors for each class
cmap = plt.cm.get_cmap(
    "tab10", np.max(class_labels) + 1
)  # 'tab10' colormap has 10 distinct colors

# TODO We need the class labels for a mapping

# Convert class labels to RGB colors using the colormap
colored_image = cmap(class_labels / np.max(class_labels))
plt.imshow(colored_image, aspect="auto")
plt.title(f"{scene} segmentation")
plt.axis("off")
# plt.legend()
# plt.show()

plt.scatter(
    pred[:, 0],
    pred[:, 1],
    c="green",
    s=scatter_size,
)
plt.scatter(
    inp[:, 0],
    inp[:, 1],
    c="purple",
    s=scatter_size,
)

# %%
# .shape
# plt.imshow(scene_image_segm)

center_x = size_unpadded[scene][0] / 2
center_y = size_unpadded[scene][1] / 2
inp_f2 = (
    rotate_around_point_torch(
        torch.from_numpy(input_trajectory[0]),
        degrees=rotation_param,
        origin=(center_x, center_y),
    )
    .cpu()
    .numpy()
)

print(center_x, center_y, inp_f2)

data = np.transpose(scene_image_segm_f.argmax(axis=1), (1, 2, 0))
class_labels = np.squeeze(data)

# Create a color map with distinct colors for each class
cmap = plt.cm.get_cmap(
    "tab10", np.max(class_labels) + 1
)  # 'tab10' colormap has 10 distinct colors

# TODO We need the class labels for a mapping

# Convert class labels to RGB colors using the colormap
colored_image = cmap(class_labels / np.max(class_labels))
plt.imshow(colored_image, aspect="auto")
plt.title(f"{scene} segmentation")
# plt.axis('off')
# plt.legend()
# plt.show()

plt.scatter(
    pred_f[:, 0],
    pred_f[:, 1],
    c="green",
    s=scatter_size,
)
plt.scatter(
    inp[:, 0],
    inp[:, 1],
    c="red",
    s=scatter_size,
)
plt.scatter(
    inp_f[:, 0],
    inp_f[:, 1],
    c="purple",
    s=scatter_size,
)
plt.scatter(
    inp_f2[:, 0],
    inp_f2[:, 1],
    c="black",
    s=scatter_size,
)

# %%
# .shape
# plt.imshow(scene_image_segm)
import torch


plt.figure()
data = np.transpose(
    rotate_tensor(scene_image_segm, 180, size_unpadded[scene])
    .argmax(axis=1)
    .cpu()
    .numpy(),
    (1, 2, 0),
)
class_labels = np.squeeze(data)

# Create a color map with distinct colors for each class
cmap = plt.cm.get_cmap(
    "tab10", np.max(class_labels) + 1
)  # 'tab10' colormap has 10 distinct colors

# TODO We need the class labels for a mapping

# Convert class labels to RGB colors using the colormap
colored_image = cmap(class_labels / np.max(class_labels))
plt.imshow(colored_image, aspect="auto")
plt.title(f"{scene} segmentation")
plt.axis("off")
# plt.legend()
# plt.show()

# %%
40 not in {30, 100, 10}

# %%
data = np.transpose(scene_image_segm.argmax(axis=1).cpu().numpy(), (1, 2, 0))
class_labels = np.squeeze(data)

# Create a color map with distinct colors for each class
cmap = plt.cm.get_cmap(
    "tab10", np.max(class_labels) + 1
)  # 'tab10' colormap has 10 distinct colors

# TODO We need the class labels for a mapping

# Convert class labels to RGB colors using the colormap
colored_image = cmap(class_labels / np.max(class_labels))
plt.imshow(colored_image, aspect="auto")
plt.title(f"{scene} segmentation")
plt.axis("off")

# %%
import imageio.v3 as iio
import matplotlib.patches as mpatches

data = iio.imread("data/SDD_semantic_maps/test_masks/little_1_mask.png")
# data = iio.imread('data/SDD_semantic_maps/test_masks/coupa_1_mask.png')
print(data.shape)
class_labels = np.squeeze(data)

num_labels = np.max(class_labels) + 1

# Create a color map with distinct colors for each class
cmap = plt.cm.get_cmap(
    "tab10", np.max(class_labels) + 1
)  # 'tab10' colormap has 10 distinct colors

# Convert class labels to RGB colors using the colormap
colored_image = cmap(class_labels / np.max(class_labels))

colors = [cmap(i / (num_labels - 1)) for i in range(num_labels)]

plt.imshow(colored_image, aspect="auto")
# plt.title(f"{scene} segmentation")

# 0 - background (blue)
# 1 - pavement (green)
# 2 - road (purple)
# 3 - structure (pink)
# 4 - terrain (yellow)
# 5 - tree (turkis/teal)

labels = ["Background", "Pavement", "Road", "Structure", "Terrain", "Tree"]
patches = [
    mpatches.Patch(color=color, label=label) for color, label in zip(colors, labels)
]

plt.legend(handles=patches, loc="upper right")

plt.axis("off")
plt.tight_layout()
plt.savefig("segmentation.pdf", dpi=400, bbox_inches="tight")
# plt.show()

# %%
(
    input_trajectory.shape,
    pred_traj.shape,
    gt_future.shape,
    pred_waypoint_map_sigmoid.shape,
)

# %%
# predict_and_evaluate(
#     model,
#     test_loader,
#     test_images,
#     num_goals,
#     num_traj,
#     obs_len=obs_len,
#     batch_size=batch_size,
#     device=device,
#     input_template=input_template,
#     waypoints=params["waypoints"],
#     resize=params["resize"],
#     temperature=params["temperature"],
#     use_TTST=True,
#     use_CWS=True if len(params["waypoints"]) > 1 else False,
#     rel_thresh=params["rel_threshold"],
#     CWS_params=params["CWS_params"],
#     dataset_name=dataset_name,
#     homo_mat=ynet_model.homo_mat,
#     mode="test",
# )

# %%
for trajectory, meta, scene in test_loader:
    # print(meta, scene)
    plt.figure()
    plt.title(scene)
    plt.imshow(test_images_orig[scene])

    for t in trajectory:
        plt.scatter(t[:, 0], t[:, 1], s=2)
        # break

# %%



