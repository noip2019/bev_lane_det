import json
import os

import cv2
import numpy as np
import torch
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

from utils.coord_util import IPM2ego_matrix


def _as_path_list(path_or_paths):
    if path_or_paths is None:
        return []
    if isinstance(path_or_paths, (list, tuple)):
        return [path for path in path_or_paths if path]
    return [path_or_paths]


def _should_log():
    rank = os.environ.get("RANK")
    return rank in (None, "", "0")


class InvalidSampleError(RuntimeError):
    pass


class ONCE3DLanesDatasetWithOffset(Dataset):
    def __init__(
        self,
        label_root,
        image_roots,
        x_range,
        y_range,
        meter_per_pixel,
        data_trans,
        output_2d_shape,
        index_file=None,
        max_samples=None,
        skip_missing_images=True,
    ):
        self.label_root = label_root
        self.image_roots = _as_path_list(image_roots)
        self.index_file = index_file
        self.x_range = x_range
        self.y_range = y_range
        self.meter_per_pixel = meter_per_pixel
        self.trans_image = data_trans
        self.output2d_size = output_2d_shape
        self.skip_missing_images = skip_missing_images
        self.max_samples = max_samples
        self.skipped_invalid_labels = 0
        self._invalid_indices = set()
        self._logged_invalid_samples = set()

        self.lane2d_thick = 3
        self.lane_length_threshold = 3.0
        self.ipm_h = int((self.x_range[1] - self.x_range[0]) / self.meter_per_pixel)
        self.ipm_w = int((self.y_range[1] - self.y_range[0]) / self.meter_per_pixel)
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(
                "No ONCE-3DLanes samples were found. Check label/image roots or provide a valid image root for this split."
            )

    def _build_samples(self):
        if self.index_file and os.path.isfile(self.index_file):
            rel_paths = self._load_index_file(self.index_file)
        else:
            rel_paths = self._scan_label_root(self.label_root)

        samples = []
        invalid_examples = []
        for rel_path in rel_paths:
            rel_path = rel_path.lstrip("/")
            if not rel_path.endswith(".jpg"):
                continue

            sequence, camera, image_name = rel_path.split("/")
            frame_name = os.path.splitext(image_name)[0]
            label_path = os.path.join(self.label_root, sequence, camera, frame_name + ".json")
            if not os.path.isfile(label_path):
                continue
            label_ok, label_issue = self._is_indexable_label_file(label_path)
            if not label_ok:
                self.skipped_invalid_labels += 1
                if len(invalid_examples) < 5:
                    invalid_examples.append(f"{label_path} ({label_issue})")
                continue

            image_path = self._resolve_image_path(rel_path)
            if image_path is None:
                if self.skip_missing_images:
                    continue
                raise FileNotFoundError(f"Image for {rel_path} not found in {self.image_roots}")

            samples.append(
                {
                    "sequence": sequence,
                    "camera": camera,
                    "frame": frame_name,
                    "image_path": image_path,
                    "label_path": label_path,
                }
            )

            if self.max_samples is not None and len(samples) >= self.max_samples:
                break

        if invalid_examples and _should_log():
            print(
                "[ONCE3DLanesDatasetWithOffset] "
                f"Skipped {self.skipped_invalid_labels} invalid label files. "
                f"Examples: {'; '.join(invalid_examples)}"
            )

        return samples

    @staticmethod
    def _load_index_file(index_file):
        with open(index_file, "r") as handle:
            return [line.strip() for line in handle if line.strip()]

    @staticmethod
    def _scan_label_root(label_root):
        rel_paths = []
        for sequence in sorted(os.listdir(label_root)):
            camera_dir = os.path.join(label_root, sequence, "cam01")
            if not os.path.isdir(camera_dir):
                continue
            for file_name in sorted(os.listdir(camera_dir)):
                if not file_name.endswith(".json"):
                    continue
                rel_paths.append(os.path.join(sequence, "cam01", file_name.replace(".json", ".jpg")))
        return rel_paths

    def _resolve_image_path(self, rel_path):
        for image_root in self.image_roots:
            candidate = os.path.join(image_root, rel_path.lstrip("/"))
            if os.path.isfile(candidate):
                return candidate
        return None

    @staticmethod
    def _is_indexable_label_file(label_path):
        try:
            if os.path.getsize(label_path) <= 0:
                return False, "empty JSON file"
        except OSError as exc:
            return False, str(exc)
        return True, None

    @staticmethod
    def _load_label(label_path):
        try:
            with open(label_path, "r") as handle:
                gt = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise InvalidSampleError(f"failed to parse {label_path}: {exc}") from exc

        if not isinstance(gt, dict):
            raise InvalidSampleError(f"label {label_path} is not a JSON object")

        missing_keys = [key for key in ("calibration", "lanes") if key not in gt]
        if missing_keys:
            raise InvalidSampleError(
                f"label {label_path} is missing required keys: {', '.join(missing_keys)}"
            )

        return gt

    def _mark_invalid_sample(self, idx, exc):
        self._invalid_indices.add(idx)
        sample = self.samples[idx]
        sample_id = sample["label_path"]
        if sample_id in self._logged_invalid_samples or not _should_log():
            return
        self._logged_invalid_samples.add(sample_id)
        print(f"[ONCE3DLanesDatasetWithOffset] Skipping invalid sample {sample_id}: {exc}")

    @staticmethod
    def _project_lane_to_image(lane_points, calibration):
        lane_points = lane_points[lane_points[:, 2] > 1e-5]
        if lane_points.shape[0] < 2:
            return None

        lane_homo = np.concatenate(
            [lane_points[:, :3], np.ones((lane_points.shape[0], 1), dtype=lane_points.dtype)],
            axis=1,
        )
        pixels = lane_homo @ calibration.T
        pixels = pixels / np.clip(pixels[:, 2:3], 1e-6, None)
        return pixels[:, :2]

    @staticmethod
    def _deduplicate_by_forward(forward, lateral, height):
        order = np.argsort(forward)
        forward = forward[order]
        lateral = lateral[order]
        height = height[order]

        unique_forward = []
        unique_lateral = []
        unique_height = []
        last_forward = None
        for fwd, lat, hgt in zip(forward, lateral, height):
            if last_forward is not None and np.isclose(fwd, last_forward):
                unique_lateral[-1] = lat
                unique_height[-1] = hgt
                continue
            unique_forward.append(fwd)
            unique_lateral.append(lat)
            unique_height.append(hgt)
            last_forward = fwd

        return (
            np.asarray(unique_forward, dtype=np.float32),
            np.asarray(unique_lateral, dtype=np.float32),
            np.asarray(unique_height, dtype=np.float32),
        )

    def get_y_offset_and_z(self, lanes_ipm):
        def lookup_offset_and_height(base_point, lane_points, lane_heights, lane_points_int):
            condition = np.where(
                (lane_points_int[0] == int(base_point[0])) & (lane_points_int[1] == int(base_point[1]))
            )
            if len(condition[0]) == 0:
                return None, None
            lane_points_selected = lane_points.T[condition]
            lane_heights_selected = lane_heights.T[condition]
            offset_y = np.mean(lane_points_selected[:, 1]) - base_point[1]
            height = np.mean(lane_heights_selected[:, 1])
            return offset_y, height

        dense_lane_points = {}
        dense_lane_heights = {}
        dense_lane_points_int = {}
        dense_lane_points_bin = {}

        for lane_idx, ipm_points_raw in lanes_ipm.items():
            ipm_points_raw = np.asarray(ipm_points_raw, dtype=np.float32)
            valid = (ipm_points_raw[1] >= 0) & (ipm_points_raw[1] < self.ipm_h)
            ipm_points = ipm_points_raw.T[valid].T
            if ipm_points.shape[1] <= 1:
                continue

            forward_rows = ipm_points[1]
            lateral_cols = ipm_points[0]
            heights = ipm_points[2]

            forward_rows, lateral_cols, heights = self._deduplicate_by_forward(
                forward_rows, lateral_cols, heights
            )
            if forward_rows.shape[0] <= 1:
                continue

            dense_forward = np.linspace(forward_rows.min(), forward_rows.max(), max(int((forward_rows.max() - forward_rows.min()) // 0.05), 2))
            dense_forward_bin = np.linspace(
                int(forward_rows.min()),
                int(forward_rows.max()),
                max(int(forward_rows.max()) - int(forward_rows.min()) + 1, 2),
            )

            if forward_rows.shape[0] <= 2:
                interp_kind = "linear"
            elif forward_rows.shape[0] == 3:
                interp_kind = "quadratic"
            else:
                interp_kind = "cubic"

            lateral_interp = interp1d(forward_rows, lateral_cols, kind=interp_kind, fill_value="extrapolate")
            height_interp = interp1d(forward_rows, heights, kind=interp_kind, fill_value="extrapolate")

            dense_lateral = lateral_interp(dense_forward)
            dense_lateral_bin = lateral_interp(dense_forward_bin)
            dense_height = height_interp(dense_forward)

            dense_lane_points[lane_idx] = np.array([dense_forward, dense_lateral], dtype=np.float32)
            dense_lane_heights[lane_idx] = np.array([dense_forward, dense_height], dtype=np.float32)
            dense_lane_points_int[lane_idx] = np.array([dense_forward, dense_lateral], dtype=np.int64)
            dense_lane_points_bin[lane_idx] = np.array([dense_forward_bin, dense_lateral_bin], dtype=np.int64)

        offset_map = np.zeros((self.ipm_h, self.ipm_w), dtype=np.float32)
        height_map = np.zeros((self.ipm_h, self.ipm_w), dtype=np.float32)
        bev_instance = np.zeros((self.ipm_h, self.ipm_w), dtype=np.uint8)

        for lane_idx, lane_bin in dense_lane_points_bin.items():
            for row, col in lane_bin.T:
                if not (0 < row < self.ipm_h and 0 < col < self.ipm_w):
                    continue
                bev_instance[row, col] = lane_idx
                offset_y, height = lookup_offset_and_height(
                    np.array([row, col]),
                    dense_lane_points[lane_idx],
                    dense_lane_heights[lane_idx],
                    dense_lane_points_int[lane_idx],
                )
                if offset_y is None:
                    bev_instance[row, col] = 0
                    continue
                offset_map[row, col] = np.clip(offset_y, 0.0, 1.0)
                height_map[row, col] = height

        return bev_instance, offset_map, height_map

    def get_seg_offset(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise InvalidSampleError(f"failed to read image {sample['image_path']}")

        gt = self._load_label(sample["label_path"])

        calibration = np.asarray(gt["calibration"], dtype=np.float32)
        image_gt = np.zeros(image.shape[:2], dtype=np.uint8)
        matrix_ipm_to_ego = IPM2ego_matrix(
            ipm_center=(int(self.x_range[1] / self.meter_per_pixel), int(self.y_range[1] / self.meter_per_pixel)),
            m_per_pixel=self.meter_per_pixel,
        )

        lanes_ipm = {}
        lane_instance_id = 1
        for lane in gt["lanes"]:
            lane_points = np.asarray(lane, dtype=np.float32)
            if lane_points.ndim != 2 or lane_points.shape[0] < 2:
                continue

            lane_points = lane_points[np.argsort(lane_points[:, 2])]
            lane_uv = self._project_lane_to_image(lane_points, calibration)
            if lane_uv is not None:
                cv2.polylines(image_gt, [lane_uv.astype(np.int32)], False, lane_instance_id, self.lane2d_thick)

            lateral = lane_points[:, 0]
            height = lane_points[:, 1]
            forward = lane_points[:, 2]
            lane_span = np.hypot(forward[-1] - forward[0], lateral[-1] - lateral[0])
            if lane_span < self.lane_length_threshold:
                continue

            ego_points = np.vstack([forward, lateral])
            ipm_points = np.linalg.inv(matrix_ipm_to_ego[:, :2]) @ (
                ego_points - matrix_ipm_to_ego[:, 2].reshape(2, 1)
            )
            ipm_points_swapped = np.zeros_like(ipm_points)
            ipm_points_swapped[0] = ipm_points[1]
            ipm_points_swapped[1] = ipm_points[0]
            lanes_ipm[lane_instance_id] = np.concatenate([ipm_points_swapped, height.reshape(1, -1)], axis=0)
            lane_instance_id += 1

        bev_gt, offset_y_map, height_map = self.get_y_offset_and_z(lanes_ipm)
        return image, image_gt, bev_gt, offset_y_map, height_map

    def __getitem__(self, idx):
        attempts = 0
        while attempts < len(self.samples):
            if idx in self._invalid_indices:
                idx = (idx + 1) % len(self.samples)
                attempts += 1
                continue
            try:
                image, image_gt, bev_gt, offset_y_map, height_map = self.get_seg_offset(idx)
                break
            except InvalidSampleError as exc:
                self._mark_invalid_sample(idx, exc)
                idx = (idx + 1) % len(self.samples)
                attempts += 1
        else:
            raise RuntimeError("Unable to fetch a valid ONCE-3DLanes sample because all candidates were invalid.")

        transformed = self.trans_image(image=image)
        image = transformed["image"]

        image_gt = cv2.resize(
            image_gt,
            (self.output2d_size[1], self.output2d_size[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        image_gt_instance = torch.tensor(image_gt).unsqueeze(0)
        image_gt_segment = torch.clone(image_gt_instance)
        image_gt_segment[image_gt_segment > 0] = 1

        bev_gt_instance = torch.tensor(bev_gt).unsqueeze(0)
        bev_gt_offset = torch.tensor(offset_y_map).unsqueeze(0)
        bev_gt_height = torch.tensor(height_map).unsqueeze(0)
        bev_gt_segment = torch.clone(bev_gt_instance)
        bev_gt_segment[bev_gt_segment > 0] = 1

        return (
            image,
            bev_gt_segment.float(),
            bev_gt_instance.float(),
            bev_gt_offset.float(),
            bev_gt_height.float(),
            image_gt_segment.float(),
            image_gt_instance.float(),
        )

    def __len__(self):
        return len(self.samples)


class ONCE3DLanesDatasetWithOffsetVal(Dataset):
    def __init__(
        self,
        label_root,
        image_roots,
        data_trans,
        index_file=None,
        max_samples=None,
        skip_missing_images=True,
    ):
        self.label_root = label_root
        self.image_roots = _as_path_list(image_roots)
        self.trans_image = data_trans
        self.index_file = index_file
        self.max_samples = max_samples
        self.skip_missing_images = skip_missing_images
        self.samples = self._build_samples()

        if not self.samples:
            raise RuntimeError(
                "No ONCE-3DLanes validation samples were found. Check label/image roots or provide a valid image root for this split."
            )

    def _build_samples(self):
        helper = ONCE3DLanesDatasetWithOffset(
            label_root=self.label_root,
            image_roots=self.image_roots,
            x_range=(0, 1),
            y_range=(0, 1),
            meter_per_pixel=1.0,
            data_trans=self.trans_image,
            output_2d_shape=(1, 1),
            index_file=self.index_file,
            max_samples=self.max_samples,
            skip_missing_images=self.skip_missing_images,
        )
        return helper.samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = cv2.imread(sample["image_path"])
        if image is None:
            raise RuntimeError(f"Failed to read image {sample['image_path']}")
        transformed = self.trans_image(image=image)
        image = transformed["image"]
        return image, sample

    def __len__(self):
        return len(self.samples)
