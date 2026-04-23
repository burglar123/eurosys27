import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
from tqdm import tqdm

from nano_pearl.utils.pearl_logger import logger


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param_data = param.data
    param_data.zero_()
    common_shape = tuple(min(a, b) for a, b in zip(param_data.shape, loaded_weight.shape))
    slices = tuple(slice(0, dim) for dim in common_shape)
    param_data[slices].copy_(loaded_weight[slices])


def _find_weight_files(path: str):
    """
    Priority:
    1. safetensors shards / single-file safetensors
    2. pytorch_model*.bin shards
    3. fallback *.bin
    """
    safetensor_files = sorted(glob(os.path.join(path, "*.safetensors")))
    if safetensor_files:
        return safetensor_files, "safetensors"

    bin_files = sorted(glob(os.path.join(path, "pytorch_model*.bin")))
    if bin_files:
        return bin_files, "bin"

    bin_files = sorted(glob(os.path.join(path, "*.bin")))
    if bin_files:
        return bin_files, "bin"

    raise FileNotFoundError(
        f"No weight files found under {path}. "
        f"Expected '*.safetensors' or 'pytorch_model*.bin'."
    )


def _iter_safetensors(file: str):
    with safe_open(file, framework="pt", device="cpu") as f:
        for weight_name in f.keys():
            yield weight_name, f.get_tensor(weight_name)


def _iter_bin(file: str):
    state_dict = torch.load(file, map_location="cpu")
    if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
        state_dict = state_dict["state_dict"]

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported .bin checkpoint format in file: {file}")

    for weight_name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        yield weight_name, tensor


def _load_one_weight(
    model: nn.Module,
    packed_modules_mapping: dict,
    weight_name: str,
    loaded_weight: torch.Tensor,
):
    # packed modules first, e.g. q_proj/k_proj/v_proj -> qkv_proj
    for k, (v, shard_id) in packed_modules_mapping.items():
        if k in weight_name:
            param_name = weight_name.replace(k, v, 1)
            param = model.get_parameter(param_name)
            weight_loader = getattr(param, "weight_loader")
            weight_loader(param, loaded_weight, shard_id)
            return

    # normal parameter
    param = model.get_parameter(weight_name)
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    files, file_format = _find_weight_files(path)

    is_master = model.lm_head.local_tp_rank == 0
    if is_master:
        logger.info(f"Loading weights from {path}")
        logger.info(f"Detected {len(files)} {file_format} file(s).")

    pbar = tqdm(files, dynamic_ncols=True, desc="Loading model", disable=not is_master)

    loaded_count = 0
    for file in pbar:
        if file_format == "safetensors":
            iterator = _iter_safetensors(file)
        elif file_format == "bin":
            iterator = _iter_bin(file)
        else:
            raise ValueError(f"Unsupported weight format: {file_format}")

        for weight_name, loaded_weight in iterator:
            try:
                _load_one_weight(model, packed_modules_mapping, weight_name, loaded_weight)
                loaded_count += 1
            except AttributeError as e:
                raise AttributeError(
                    f"Failed to load weight '{weight_name}' from file '{file}'. "
                    f"This usually means the target parameter does not have the expected "
                    f"weight_loader signature."
                ) from e
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load weight '{weight_name}' from file '{file}'."
                ) from e

    if loaded_count == 0:
        raise RuntimeError(
            f"Found weight files under {path}, but no tensor weights were loaded."
        )

    if is_master:
        logger.info(f"Finished loading model weights. Total tensors loaded: {loaded_count}")