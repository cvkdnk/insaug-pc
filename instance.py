import os
import numpy as np
import yaml
from pathlib import Path
import threading


class KittiInstance:
    def __init__(self, save_class, config, kitti_yaml, seq=None):
        if not seq:
            seq = range(11)
        for i in range(len(seq)):
            seq[i] = str(seq[i]).zfill(2)
        learning_map = kitti_yaml["learning_map"]
        trans_label = kitti_yaml["labels"]
        inv_map = kitti_yaml["learning_map_inv"]
        ins_save_path = Path(config["instance_lib"])
        data_path = Path(config["data_path"])
        for i in save_class:
            ins_save_path.joinpath(trans_label[inv_map[i]]+"_"+str(i).zfill(2)).mkdir(exist_ok=True)

        for sequence in os.listdir(data_path):
            sequence = str(sequence)
            if sequence in seq:
                data_seq = data_path.joinpath(sequence)
                velodyne_path = data_seq.joinpath("velodyne")
                length = len(os.listdir(velodyne_path))
                for idx, frame_file in enumerate(os.listdir(velodyne_path)):
                    frame_file = str(frame_file)
                    # read data
                    data = np.fromfile(
                        velodyne_path.joinpath(frame_file),
                        dtype=np.float32
                    ).reshape((-1, 4))[:, :3]
                    labels = np.fromfile(
                        str(velodyne_path).replace('velodyne', 'labels') +"/"+ frame_file[:-3] + 'label',
                        dtype=np.uint32
                    )
                    sem_labels = labels & 0xFFFF
                    ins_labels = labels >> 16
                    sem_labels = np.vectorize(learning_map.__getitem__)(sem_labels)
                    for cls in save_class:
                        save_path = ins_save_path.joinpath(trans_label[inv_map[cls]]+"_"+str(cls).zfill(2))
                        select_idx = self.select_cls(sem_labels, cls)
                        for idx2, ins in enumerate(self.get_ins(ins_labels[select_idx], data[select_idx])):
                            level = self.level(ins)
                            filename = sequence+frame_file[:-4]+str(idx2)+"_"+str(ins.shape[0]).zfill(4)+"_"+str(level).\
                                zfill(2)+".npy"
                            # normalize
                            ins[:, 2] -= np.min(ins[:, 2])
                            ins[:, :2] -= np.mean(ins[:, :2], axis=0)
                            np.save(str(save_path.joinpath(filename)), ins)
                    print("sequence{}: {}/{}".format(sequence, idx+1, length))

    @staticmethod
    def select_cls(sem_labels, cls):
        select_idx = (sem_labels == cls).nonzero()
        return select_idx

    @staticmethod
    def get_ins(ins_labels, points):
        labels = np.unique(ins_labels)
        for label in labels:
            yield points[(ins_labels == label).nonzero()]

    @staticmethod
    def save_xyz(points, save_path):
        for idx, point in enumerate(points):
            with open(save_path, 'w') as f:
                for idx, point in enumerate(point):
                    string = str(point[0].item()) + ',' + str(point[1].item()) + ',' + str(point[2].item())
                    if idx != len(point) - 1:
                        string += '\n'
                    f.write(string)

    @staticmethod
    def level(ins):
        center = np.mean(ins[:, :2], axis=0)
        distance = np.sqrt(np.sum(center**2))
        return int(distance)

if __name__ == "__main__":
    save_cls = [2, 3, 4, 5, 6, 7, 8, 14, 18, 19]
    t = [threading.Thread(target=KittiInstance, args=(save_cls, [0, 1])),
         threading.Thread(target=KittiInstance, args=(save_cls, [2, 3])),
         threading.Thread(target=KittiInstance, args=(save_cls, [4, 5])),
         threading.Thread(target=KittiInstance, args=(save_cls, [6, 7])),
         threading.Thread(target=KittiInstance, args=(save_cls, [9, 10]))]
    for i in t:
        i.start()
    for i in t:
        i.join()
