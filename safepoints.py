# -*- coding: UTF-8 -*-

import numpy as np
import yaml
import os
from collections import defaultdict
import pickle
import logging
from pathlib import Path
from tqdm import tqdm
from instance import KittiInstance
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch


# ---------      /    /_     ___  /____
#   __   |      /    /\/       | / \  /
#   |_|  |     /| |   /\     ---    \/
#        |      | |  / /\    | /    /\
#       \|      |      //    |/    /  \

GROUP_DIST = 5  # 每5m分为一组
THRESHOLD_RATIO = 25  # 保留上75%的点


SAFE_RADIUS = {
    2: 2,     # bicycle
    7: 2,     # bicyclist
    14: 0,    # fence
    3: 2,     # motorcycle
    8: 2,     # motorcyclist
    5: 5,     # other vehicle
    6: 2,   # person
    18: 2,  # pole
    19: 2,    # traffic sign
    4: 5      # truck
}

GROUND_CLASS = [
    9,  # road
    10,  # parking
    11,  # sidewalk
    12  # other-ground
]


def set_log_infile(log_path):
    logger = logging.getLogger("AugmentKITTI")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s] %(message)s', "%H:%M:%S")
    file_handle = logging.FileHandler(log_path)
    file_handle.setLevel(logging.INFO)
    file_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    stream_handle = logging.StreamHandler()
    stream_handle.setLevel(logging.INFO)
    stream_handle.setFormatter(formatter)
    logger.addHandler(file_handle)
    logger.addHandler(stream_handle)
    return logger

class AugmentKITTI:
    def __init__(self, log_path, config, kitti_yaml):
        self.logger = set_log_infile(log_path)
        self.kitti_yaml = kitti_yaml
        self.data_path = config["data_path"]
        self.instance_lib = config["instance_lib"]
        self.lib_dict = {dirname: int(dirname[-2:]) for dirname in os.listdir(self.instance_lib)}
        self.select_ins_idx = {}  # {label: {dist_lv: [index in self.ins_lib_dict]}}
        self.augment_seq = [0, 1, 2, 3, 4, 5, 6, 7, 9, 10]
        self.data = []
        self.data_seq_frame = []
        self._load_data_path()
        self.ins_lib_dict = {}  # {label: {"filename": [], "distance": [], "points_num": []}} 按索引一一对应
        self.points = None
        self.sem_labels = None
        self.ins_labels = None
        self.safe_radius = list(SAFE_RADIUS.values())
        self.safe_radius = list(-np.sort(-np.unique(np.array(self.safe_radius, dtype=np.float32))))
        if 0 in self.safe_radius:
            self.safe_radius.remove(0)
        self.reset()

    def reset(self):
        self.points = None
        self.sem_labels = None
        self.ins_labels = None

    def _load_data(self, bin_path):
        self.reset()
        self.points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
        label_path = bin_path.replace("velodyne", "labels")[:-3] + "label"
        labels = np.fromfile(label_path, dtype=np.uint32)
        self.sem_labels = labels & 0xFFFF
        self.ins_labels = labels >> 16
        learning_map = self.kitti_yaml["learning_map"]
        self.sem_labels = np.vectorize(learning_map.__getitem__)(self.sem_labels)

    def _load_data_path(self):
        for seq_dirname in os.listdir(self.data_path):
            if int(seq_dirname) in self.augment_seq:
                bin_path = os.path.join(self.data_path, seq_dirname, "velodyne")
                for file_name in os.listdir(bin_path):
                    self.data.append(os.path.join(bin_path, file_name))
                    self.data_seq_frame.append((seq_dirname, file_name[:6]))
        self.logger.info("Total data: " + str(len(self.data)))

    def _load_ins_lib_path(self):
        for ins_dirname in self.lib_dict:
            ins_path = os.path.join(self.instance_lib, ins_dirname)
            label = self.lib_dict[ins_dirname]
            self.ins_lib_dict[label] = {"filename":[], "distance":[], "points_num":[]}
            for filename in os.listdir(ins_path):
                distance = int(filename[-6:-4])
                points_num = int(filename[-11:-7])
                self.ins_lib_dict[label]["filename"].append(filename)
                self.ins_lib_dict[label]["distance"].append(distance)
                self.ins_lib_dict[label]["points_num"].append(points_num)
            print(ins_dirname, "has loaded already, total {} instances".format(
                len(self.ins_lib_dict[label]["filename"])
            ))

    def _ins_lib_filter(self):
        """筛选instance, 将符合条件的instance索引保存在self.select_ins_idx中

        可以改动的地方：目前是距离传感器每 *5m* 划分为一组，每组根据instance的点数量，仅保留上 *75%* 的instance（筛选掉被遮挡过多的实例）
        """
        for label in self.ins_lib_dict:
            select_idx = defaultdict(list)
            select_points_num = defaultdict(list)
            for idx, dist in enumerate(self.ins_lib_dict[label]["distance"]):
                select_idx[dist//GROUP_DIST].append(idx)
                select_points_num[dist//GROUP_DIST].append(self.ins_lib_dict[label]["points_num"][idx])
            select_idx = dict(select_idx)
            select_points_num = dict(select_points_num)
            for dist_lv in dict(select_points_num):
                select_idx[dist_lv] = np.asarray(select_idx[dist_lv])
                select_points_num_np = np.asarray(select_points_num[dist_lv])
                threshold = np.percentile(select_points_num_np, THRESHOLD_RATIO)
                reserve = (select_points_num_np > threshold).nonzero()
                select_idx[dist_lv] = select_idx[dist_lv][reserve]
            self.select_ins_idx[label] = select_idx

    def _save_select_instance(self, save_path):
        if self.select_ins_idx == {}:
            print("没有实例路径需要保存")
            return
        save_dict = {}  # {label: {dist_lv: [filepath...]}}
        self.logger.info("Save instance number:")
        for label in self.select_ins_idx:
            save_dict[label] = defaultdict(list)
            for dist_lv in self.select_ins_idx[label]:
                np_idx = self.select_ins_idx[label][dist_lv]
                self.logger.info(
                    "label{} distance {}~{}：{}".format(
                        label,
                        dist_lv*GROUP_DIST,
                        (dist_lv+1)*GROUP_DIST,
                        np_idx.shape[0]
                    )
                )
                for idx in np_idx:
                    save_dict[label][dist_lv].append(
                        self.ins_lib_dict[label]["filename"][idx.item()]
                    )
        pickle.dump(dict(save_dict), open(save_path, 'wb'))

    def save_ins_pkl(self, save_path):
        self.logger.info("================================================")
        self.logger.info("SAVING INSTANCE")
        self.logger.info("loading instance lib path")
        self._load_ins_lib_path()
        self.logger.info("filtering the instance lib")
        self._ins_lib_filter()
        self.logger.info("saving the select instances")
        self._save_select_instance(save_path)
        self.logger.info("complete")

    @staticmethod
    def load_ins_pkl(load_path):
        """用法：如找到person中10m-15m实例，load_dict[label][3]得到list，再从list中选取即可"""
        load_dict = pickle.load(open(load_path, 'rb'))  # {label:{dist_lv:[filename...]}}
        return load_dict

    def _select_safe_points(self, save_path, seq, frame):
        assert self.points is not None, "Need load data before select_p."
        object_idx = np.arange(self.sem_labels.shape[0])
        for i in GROUND_CLASS:
            object_idx = np.intersect1d((self.sem_labels != i).nonzero(), object_idx)
        ground_idx = np.setdiff1d(np.arange(self.sem_labels.shape[0]), object_idx)
        # random_idx = np.random.permutation(ground_idx.shape[0])
        # ground_idx = ground_idx[random_idx]
        object_points = torch.tensor(self.points[object_idx, :2][::4], dtype=torch.float32).cuda()  # CUDA
        safe_points = {r: {} for r in self.safe_radius}
        raw_ground_points = self.points[ground_idx]
        ground_points = torch.tensor(self.points[ground_idx, :2], dtype=torch.float32).cuda()  # CUDA
        ng = ground_points.shape[0]
        no = object_points.shape[0]
        ground_points_ = ground_points.contiguous().unsqueeze(1).expand(ng, no, 2)
        dist_mat_square = ((ground_points_ - object_points) ** 2).sum(dim=2)
        min_dist_square = dist_mat_square.min(dim=1)[0]
        distance_mat = (ground_points ** 2).sum(dim=1)
        points_in_50m = distance_mat < 50
        for r in safe_points:
            safe_index = ((min_dist_square > r**2) & points_in_50m).nonzero().cpu().numpy().reshape(-1)
            raw_safepoints = raw_ground_points[safe_index]
            distlv = np.sqrt(np.sum(raw_safepoints[:, :2]**2, axis=1)) // 5
            for i in range(20):
                safe_points[r][i] = raw_safepoints[distlv==i]
                _, grid_sample = np.unique(
                    safe_points[r][i]//1,
                    return_index=True,
                    axis=0
                )
                safe_points[r][i] = safe_points[r][i][grid_sample]
        # for p_idx in ground_idx:
        #     dist_square = ((object_points[:, :2] - self.points[p_idx, :2]) ** 2).sum(dim=1)
        #     for r in safe_points:
        #         if dist_square.min() > r**2:
        #             safe_points[r].append(self.points[p_idx])

        """Save in file"""
        with open(save_path, 'wb') as f:
            pickle.dump(safe_points, f)

    def save_safe_points(self, save_path):
        self.logger.info("================================================")
        self.logger.info("SAVING SAFE POINTS")
        for seq in self.augment_seq:
            create_dir = Path(os.path.join(save_path, str(seq).zfill(2)))
            create_dir.mkdir(exist_ok=True)
        pbar = tqdm(total=len(self.data))
        for data, info in zip(self.data, self.data_seq_frame):
            seq, frame = info
            self._load_data(data)
            save_dir = os.path.join(save_path, seq+"_"+frame+".pkl")
            self._select_safe_points(save_dir, seq, frame)
            pbar.update(1)
        pbar.close()


if __name__ == "__main__":
    am = AugmentKITTI("./augment_log.txt")
    am.save_safe_points("./safepoints")













