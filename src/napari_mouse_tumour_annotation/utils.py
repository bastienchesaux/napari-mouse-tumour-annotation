import inspect
import json

import numpy as np
import torch
from huggingface_hub import hf_hub_download, list_repo_files
from monai.transforms import AsDiscrete, Compose, KeepLargestConnectedComponent
from skimage.measure import block_reduce

from . import architectures

HF_REPO = "bchesaux/napari-mouse-tumour-annotation"


def full_scan_normalize(image, clip_percentile=96):
    ub = np.percentile(image, clip_percentile)
    lb = image.min()

    normalized = (image - lb) / (ub - lb)
    normalized = np.clip(normalized, 0, 1)
    return normalized


def load_model(
    model_name, checkpoint_path, device=None, deep_supervision=False
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    func = getattr(architectures, model_name)

    if "deep_supervision" in inspect.signature(func).parameters:
        model = func(deep_supervision=deep_supervision).to(device)
    else:
        model = func().to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    return model


def scan_hf_repo():
    files = list_repo_files(repo_id=HF_REPO)
    return [f.replace(".pt", "") for f in files if f.endswith(".pt")]


def load_model_hf(model_key, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = hf_hub_download(
        repo_id=HF_REPO, filename=f"{model_key}.pt"
    )
    config_path = hf_hub_download(
        repo_id=HF_REPO, filename=f"{model_key}.json"
    )

    with open(config_path) as f:
        config = json.load(f)

    func = getattr(architectures, config["model_name"])
    if "deep_supervision" in inspect.signature(func).parameters:
        model = func(deep_supervision=config["deep_supervision"]).to(device)
    else:
        model = func().to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def build_post_transform():
    post_trans = Compose(
        [
            AsDiscrete(threshold=0.5),
            KeepLargestConnectedComponent(applied_labels=[1]),
        ]
    )

    return post_trans


def extract_tumor_window(img, win_center, half_win):

    img_padded = np.pad(img, half_win, mode="constant", constant_values=0)

    # center is shifted by half because of padding
    z, y, x = (c + half_win for c in win_center)

    img_win = img_padded[
        z - half_win : z + half_win,
        y - half_win : y + half_win,
        x - half_win : x + half_win,
    ]

    return img_win


def downsize_window(img_win, factor: int):
    return block_reduce(img_win, block_size=factor)


def up_sample_pred(pred_win, factor):
    pred_win = np.repeat(pred_win, factor, axis=0)
    pred_win = np.repeat(pred_win, factor, axis=1)
    pred_win = np.repeat(pred_win, factor, axis=2)

    return pred_win


def single_image_prediction(model, image, post_trans, device):
    model.eval()

    image_tensor = (
        torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)
    )

    with torch.no_grad():
        output = model(image_tensor)
    if isinstance(output, list):
        output = output[-1]
    output = torch.sigmoid(output)
    binary = post_trans(output)

    binary = binary.cpu().numpy().astype(bool)
    return binary.squeeze()


def insert_patch(patch_pred, coords, shape, half, dtype=np.uint8):
    prediction = np.zeros(shape, dtype)

    z, y, x = coords

    z0, z1 = z - half, z + half
    y0, y1 = y - half, y + half
    x0, x1 = x - half, x + half

    cz0, cz1 = np.clip([z0, z1], 0, shape[0])
    cy0, cy1 = np.clip([y0, y1], 0, shape[1])
    cx0, cx1 = np.clip([x0, x1], 0, shape[2])

    prediction[cz0:cz1, cy0:cy1, cx0:cx1] = patch_pred[
        cz0 - z0 : cz1 - z0,
        cy0 - y0 : cy1 - y0,
        cx0 - x0 : cx1 - x0,
    ]

    return prediction


def add_new_label(labels, binary_window, win_center, win_size):
    half = win_size // 2

    labels_padded = np.pad(labels, half, mode="constant", constant_values=0)

    z, y, x = win_center + half

    labels_padded[
        z - half : z + half + win_size % 2,
        y - half : y + half + win_size % 2,
        x - half : x + half + win_size % 2,
    ] = (labels_padded.max() + 1) * binary_window

    labels_updated = labels_padded[half:-half, half:-half, half:-half]
    return labels_updated
