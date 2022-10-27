import numpy as np
import pickle
from pathlib import Path
import random
import yaml

work_path = "/home/neu-wang/cvkdnk/workspace/dianyun/cutmix"
work_path = Path(work_path)

with open(work_path + "/path.yaml", 'r') as f:
    config = yaml.safe_load(f)

ins_path = config["instance_lib"]
ins_pkl = config["ins_pkl"]
safe_points_path = config["safepoints"]

AUGMENT_NUM = config["augment_num"]
AUGMENT_WEIGHT = config["augment_weight"]

SAFE_RADIUS = {
    2: 2,  # bicycle
    7: 2,  # bicyclist
    3: 2,  # motorcycle
    8: 2,  # motorcyclist
    5: 5,  # other vehicle
    6: 0.5,  # person
    18: 0.5,  # pole
    4: 5  # truck
}

AUGMENT_DICT = {2: [2, 7, 3, 8, 6, 18], 5: [5, 4]}

INS_LIB_DIRNAME = {
    2: "bicycle_02",
    7: "bicyclist_07",
    3: "motorcycle_03",
    8: "motorcyclist_08",
    5: "other-vehicle_05",
    6: "person_06",
    18: "pole_18",
    4: "truck_04"
}


class CutmixAugment:
    """Initializing this object in dataset and calling ca.cutmix() generates augment points and labels in __getitem__.
    It takes about 0.1s. Then concat the return value with semantic kitti data.
    """
    def __init__(
            self,
            rot=True,
            scale=True,
            move=True
    ):
        with open(ins_pkl, 'rb') as f:
            self.ins_dict = pickle.load(f)  # {label: {dist_lv: [filepath...]}}
        self.ins_path = Path(ins_path)
        self.aug_random = {2: [], 5: []}
        for r in self.aug_random:
            for k, v in AUGMENT_WEIGHT[r].items():
                for i in range(v):
                    self.aug_random[r].append(k)
            self.aug_random[r] = np.array(self.aug_random[r], dtype=np.int32)
        self.rot = rot
        self.scale = scale
        self.move = move

    def cutmix(self, seq: str, frame: str):
        filename = seq + "_" + frame + ".pkl"
        safe_points_dict = pickle.load(open(safe_points_path.joinpath(filename), 'rb'))
        mix_points = []
        mix_labels = []
        selected = None
        for r in safe_points_dict:
            if safe_points_dict[r] != {}:
                safe_points = []
                for j in range(AUGMENT_NUM[r]):
                    rand_distlv = np.random.randint(20)
                    if safe_points_dict[r][rand_distlv].shape[0] != 0:
                        safe_points.append(
                            safe_points_dict[r][rand_distlv][
                                np.random.randint(safe_points_dict[r][rand_distlv].shape[0])
                            ]
                        )
                safe_points = self.filter_select_points(safe_points, r, selected)
                if len(safe_points) != 0:
                    safe_points = self.filter_select_points(safe_points, r, selected)
                    if selected is None:
                        selected = safe_points
                    else:
                        selected = np.concatenate((selected, safe_points), axis=0)
                    for safe_point in safe_points.reshape((-1, 4)):
                        rand_label = int(np.random.choice(self.aug_random[r]))
                        dist_lv = int(np.sqrt((safe_point.reshape(-1)[:2] ** 2).sum()) // 5)
                        if dist_lv in self.ins_dict[rand_label]:
                            ins = self.ins_dict[rand_label][dist_lv]
                            rand_idx = random.randrange(len(ins) - 1)
                            ins = self.ins_path.joinpath(INS_LIB_DIRNAME[rand_label], ins[rand_idx])
                            ins = np.load(str(ins)).reshape(-1, 3)
                            ins = np.concatenate((ins, np.ones((ins.shape[0], 1), dtype=np.float32)), axis=1)
                            ins = np.matmul(ins, self.trans_mat(self.rot, self.scale, self.move, self.move))[:, :3]
                            ins += safe_point.reshape(-1)[:3]
                            mix_points.append(ins)
                            mix_labels.append(rand_label)
        if len(mix_points) == 0:
            return None
        elif len(mix_points) == 1:
            mix_labels = np.ones(mix_points[0].reshape(-1, 3).shape[0], dtype=np.int32) * mix_labels[0]
            return mix_points[0].reshape(-1, 3), mix_labels
        else:
            mix = mix_points[0].reshape((-1, 3))
            labels = np.ones(mix_points[0].reshape(-1, 3).shape[0], dtype=np.int32) * mix_labels[0]
            for idx, i in enumerate(mix_points[1:]):
                labels_ = np.ones(i.reshape(-1, 3).shape[0], dtype=np.int32) * mix_labels[idx + 1]
                labels = np.concatenate((labels, labels_))
                mix = np.concatenate((mix, i.reshape(-1, 3)), axis=0)
            return mix, labels

    @staticmethod
    def filter_select_points(select, safe_radius, selected=None):
        reserve = select[0].reshape(-1, 4)
        for point in select[1:]:
            dist = ((reserve[:, :2] - point[:2]) ** 2).sum(axis=1)
            if selected is not None:
                dist = np.concatenate((dist, ((selected[:, :2] - point[:2]) ** 2).sum(axis=1)))
            if dist.min() >= safe_radius ** 2:
                reserve = np.concatenate((reserve, point.reshape(-1, 4)), axis=0)
        return reserve

    @staticmethod
    def trans_mat(
            rot=False, scale=False,
            move_xy=False,
            move_z=False,
            angle=(0, 2 * np.pi),
            scale_ratio=(0.9, 1.1),
            move_range=(-0.01, 0.01)
    ):
        mat = np.diag([1.0, 1.0, 1.0, 1.0])
        if rot:
            theta = np.random.random() * (angle[1] - angle[0]) + angle[0]
            mat[0, 0] = mat[1, 1] = np.cos(theta)
            mat[0, 1] = np.sin(theta)
            mat[1, 0] = -mat[0, 1]
        if scale:
            scale_ratio = np.random.random(3) * (scale_ratio[1] - scale_ratio[0]) + scale_ratio[0]
            mat[0, 0] *= scale_ratio[0]
            mat[1, 1] *= scale_ratio[1]
            mat[2, 2] *= scale_ratio[2]
        if move_xy:
            move_dist = np.random.random(2) * (move_range[1] - move_range[0]) + move_range[0]
            mat[3, 0] += move_dist[0]
            mat[3, 1] += move_dist[1]
        if move_z:
            mat[3, 2] += np.random.random() * 0.001
        return mat


if __name__ == "__main__":
    ca = CutmixAugment()
    points, labels = ca.cutmix("00", "000000")
