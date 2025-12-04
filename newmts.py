# %%
import imageio.v3 as iio
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum


def plot_map(class_labels, class_labels2=None):
    num_labels = 6  # np.max(class_labels) + 1
    max_label = num_labels - 1

    # Create a color map with distinct colors for each class
    cmap = plt.get_cmap("tab10", num_labels)  # 'tab10' colormap has 10 distinct colors

    # Convert class labels to RGB colors using the colormap
    colored_image = cmap(class_labels / max_label)

    colors = [cmap(i / (num_labels - 1)) for i in range(num_labels)]

    if class_labels2 is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        colored_image2 = cmap(class_labels2 / max_label)
        ax1.imshow(colored_image, aspect="auto")
        ax2.imshow(colored_image2, aspect="auto")
        ax1.set_title("Original")
        ax2.set_title("Modified")
        ax1.axis("off")
        ax2.axis("off")
    else:
        plt.figure()
        plt.imshow(colored_image, aspect="auto")

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

    if class_labels2 is not None:
        fig.legend(handles=patches, loc="center right")
    else:
        plt.legend(handles=patches, loc="upper right")

    plt.axis("off")
    plt.tight_layout()
    # plt.savefig("segmentation.pdf", dpi=400, bbox_inches="tight")
    plt.show()


# data = iio.imread("data/SDD_semantic_maps/test_masks/little_1_mask.png")
data = iio.imread("data/SDD_semantic_maps/test_masks/coupa_1_mask.png")

# data = np.random.randint(0, 5 + 1, size=(10, 10))

print(data.shape)
class_labels = np.squeeze(data)

# %%
from utils.dataloader import SceneDataset, scene_collate
from torch.utils.data import DataLoader
import pandas as pd
from utils.preprocessing import create_images_dict

TEST_DATA_PATH = "data/SDD/test_trajnet.pkl"

df = pd.read_pickle(TEST_DATA_PATH)
## Add obstacle to the map
df.head()
# Needs a trajectory

plt.figure()
# plt.title(scene)
plt.imshow(data)

for tid in [28]: #df.trackId.unique():
    t = df[df.trackId == tid]
    plt.scatter(t.x, t.y, s=2)

# %%
