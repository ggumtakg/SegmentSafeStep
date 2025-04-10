from .cityscapes import CitySegmentation
from .indobohaeng import SurfaceMask

datasets = {
    'citys': CitySegmentation,
    'surface': SurfaceMask
}


def get_segmentation_dataset(name, **kwargs):
    """Segmentation Datasets"""
    dataset_root = kwargs.get("root", "/home/segmentsafestep/Fast-SCNN-pytorch-master/datasets")  # 기본 경로 설정
    kwargs["root"] = dataset_root  # root 값 명시적으로 전달
    return datasets[name.lower()](**kwargs)
