from instance import KittiInstance
from safepoints import AugmentKITTI
import yaml
import os

config = yaml.safe_load(open("./config.yaml", "r"))
kitti_yaml = yaml.safe_load(open(config["kitti_yaml"], "r"))
save_cls = [2, 3, 4, 5, 6, 7, 8, 14, 18, 19]
ki = KittiInstance(save_cls, config, kitti_yaml)
ak = AugmentKITTI("./log.txt", config, kitti_yaml)
ak.save_ins_pkl("./instance.pkl")
ak.save_safe_points("./safepoints")
work_path = os.path.abspath('.')
with open("./path.yaml", "a", encoding="utf-8") as f:
    f.write("# 自动生成的路径：\n")
    f.write("ins_pkl: " + work_path + "/instance.pkl\n")
    f.write("safepoints: " + work_path + "./safepoints\n")

