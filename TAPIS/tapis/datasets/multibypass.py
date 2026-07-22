#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import re
import logging
import numpy as np

from copy import deepcopy
from .surgical_dataset import SurgicalDataset
from . import utils as utils
from .build import DATASET_REGISTRY

logger = logging.getLogger(__name__)


@DATASET_REGISTRY.register()
class Multibypass(SurgicalDataset):
    """
    MultiBypass-4C-T40 dataloader.
    """

    def __init__(self, cfg, split, load=True):
        self.dataset_name = "MultiBypass"
        self.zero_fill = 6
        self.image_type = "jpg"
        super().__init__(cfg, split, load)

    def keyframe_mapping(self, video_idx, sec_idx, sec):
        # Annotation frame_num already matches the 0-indexed position of the
        # frame in videos_cutmargin512/<video>/, the same numbering used to
        # build frame_lists/*.csv -- no fps rescaling needed, unlike GraSP.
        return sec

    def frame_name_spliting(self, video_name, sec):
        m = re.match(r"C(\d+)V(\d+)$", video_name)
        center, video = m.groups()
        video_num = int(center) * 100 + int(video)
        return [video_num, sec]

    def frame_num_joining(self, video_num, sec):
        return self.frame_name_joining(video_num, sec)

    def frame_name_joining(self, video_name, sec):
        return f"{video_name}/{sec:0{self.zero_fill}d}.{self.image_type}"

    def __getitem__(self, idx):
        """
        Generate corresponding clips, boxes, labels and metadata for given idx.

        Args:
            idx (int): the video index provided by the pytorch sampler.
        Returns:
            frames (tensor): the frames of sampled from the video. The dimension
                is `channel` x `num frames` x `height` x `width`.
            label (ndarray): the label for correspond boxes for the current video.
            idx (int): the video index provided by the pytorch sampler.
            extra_data (dict): a dict containing extra data fields, like "boxes",
                "ori_boxes" and "metadata".
        """
        # Get the path of the middle frame
        video_idx, sec_idx, sec, center_idx = self._keyframe_indices[idx]
        video_name = self._video_idx_to_name[video_idx]
        complete_name = self.frame_name_joining(video_name, sec)

        assert self._image_paths[video_idx][center_idx].endswith(complete_name), \
            f'Different paths {complete_name} & {self._image_paths[video_idx][center_idx]}'

        # Get the frame idxs for current clip.
        if self._video_length > 1:
            seq = utils.get_sequence(
                center_idx,
                self._seq_len // 2,
                self._sample_rate,
                num_frames=len(self._image_paths[video_idx]),
            )
        else:
            seq = [center_idx]

        assert center_idx in seq, f'Center index {center_idx} not in sequence {seq}'
        clip_label_list = deepcopy(self._keyframe_boxes_and_labels[video_idx][sec_idx])
        assert len(clip_label_list) > 0

        # No region tasks for MultiBypass phases -- this dataset only supports
        # frame-level tasks (REGIONS.ENABLE is expected to be False).
        all_labels = {task: [] for task in self._region_tasks}

        for task in self._frame_tasks:
            assert all(label[task] == clip_label_list[0][task] for label in clip_label_list), \
                f'Inconsistent {task} labels for frame {complete_name}: {[label[task] for label in clip_label_list]}'
            all_labels[task] = clip_label_list[0][task]

        boxes = np.zeros((1, 4))
        extra_data = {}

        # Load images of current clip.
        image_paths = [self._image_paths[video_idx][frame] for frame in seq]
        imgs = utils.retry_load_images(
            image_paths, backend=self.cfg.ENDOVIS_DATASET.IMG_PROC_BACKEND
        )

        # Preprocess images and boxes
        imgs, boxes, _ = self._images_and_boxes_preprocessing_cv2(
            imgs, boxes=boxes
        )

        imgs = utils.pack_pathway_output(self.cfg, imgs)

        if self.cfg.NUM_GPUS > 1:
            frame_identifier = self.frame_name_spliting(video_name, sec)
        else:
            frame_identifier = complete_name

        return imgs, all_labels, extra_data, frame_identifier
