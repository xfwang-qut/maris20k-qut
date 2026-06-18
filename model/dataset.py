from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS
# from mmseg.registry import DATASETS
# from mmseg.datasets.basesegdataset import BaseSegDataset

@DATASETS.register_module()
class MarIS20K(CocoDataset):
    METAINFO = {
        'classes': ["fish", "echinus", "holothurian", "turtle", "snake", "squid", "octopus", "conch",
                    "ray", "scallop", "seahorse", "starfish", "jellyfish", "diver", "cetaceans"],
        'palette': [(220, 25, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (0, 0, 230)]
    }


@DATASETS.register_module()
class UIISDataset(CocoDataset):
    METAINFO = {
        'classes': ['fish', 'reefs', 'aquatic plants', 'wrecks/ruins', 'human divers', 'robots', 'sea-floor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }


@DATASETS.register_module()
class UIIS10KDataset(CocoDataset):
    METAINFO = {
        'classes': ['fish', 'reptiles', 'arthropoda', 'corals', 'mollusk', 'plants', 'ruins', 'garbage', 'human', 'robots'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (220, 20, 60), (255, 0, 0), (0, 0, 142),]
    }


@DATASETS.register_module()
class USIS10KDataset(CocoDataset):
    METAINFO = {
        'classes': ['wrecks/ruins', 'fish', 'reefs', 'aquatic plants', 'human divers', 'robots', 'sea-floor'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
                    (0, 60, 100), (0, 80, 100), (0, 0, 230)]
    }


@DATASETS.register_module()
class UW10KDataset(CocoDataset):
    METAINFO = {
        'classes': ['Fish', 'Sea urchins', 'Sea cucumber', 'Sea turtle', 'Sea snake', 'Squid',
                    'Octopus', 'Shrimp', 'Ray', 'Shellfish', 'Seahorse', 'Starfish', 'Jellyfish',
                    'Diver', 'Coral'],
        'palette': [(220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100),
                    (0, 80, 100), (0, 0, 230), (220, 20, 60), (255, 0, 0), (0, 0, 142),
                    (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (0, 0, 230)]
    }

@DATASETS.register_module()
class FishDataset(CocoDataset):
    METAINFO = {
        'classes': ['Fish'],
        'palette': [(220, 20, 60)]
    }


#
# @DATASETS.register_module()
# class MarIS20KSeg(BaseSegDataset):
#     METAINFO = dict(
#         classes=("fish", "echinus", "holothurian", "turtle", "snake", "squid", "octopus", "conch",
#                     "ray", "scallop", "seahorse", "starfish", "jellyfish", "diver", "cetaceans"),
#         palette=[(220, 25, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
#                 (0, 60, 100), (0, 80, 100), (0, 0, 230), (220, 20, 60), (255, 0, 0), (0, 0, 142),
#                 (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (0, 0, 230)])
#
#     def __init__(self,
#                  img_suffix='.jpg',
#                  seg_map_suffix='.png',
#                  reduce_zero_label=False,
#                  **kwargs) -> None:
#         super().__init__(
#             img_suffix=img_suffix,
#             seg_map_suffix=seg_map_suffix,
#             reduce_zero_label=reduce_zero_label,
#             **kwargs)



#
#
# @DATASETS.register_module()
# class CoralscapesDataset(BaseSegDataset):
#
#     METAINFO = dict(
#         classes=(
#             # id: 1..39 (raw ids)
#             'seagrass',                  # 1
#             'trash',                     # 2
#             'other coral dead',          # 3
#             'other coral bleached',      # 4
#             'sand',                      # 5
#             'other coral alive',         # 6
#             'human',                     # 7
#             'transect tools',            # 8
#             'fish',                      # 9
#             'algae covered substrate',   # 10
#             'other animal',              # 11
#             'unknown hard substrate',    # 12
#             'background',                # 13
#             'dark',                      # 14
#             'transect line',             # 15
#             'massive/meandering bleached',  # 16
#             'massive/meandering alive',     # 17
#             'rubble',                       # 18
#             'branching bleached',           # 19
#             'branching dead',               # 20
#             'millepora',                    # 21
#             'branching alive',              # 22
#             'massive/meandering dead',      # 23
#             'clam',                         # 24
#             'acropora alive',               # 25
#             'sea cucumber',                 # 26
#             'turbinaria',                   # 27
#             'table acropora alive',         # 28
#             'sponge',                       # 29
#             'anemone',                      # 30
#             'pocillopora alive',            # 31
#             'table acropora dead',          # 32
#             'meandering bleached',          # 33
#             'stylophora alive',             # 34
#             'sea urchin',                   # 35
#             'meandering alive',             # 36
#             'meandering dead',              # 37
#             'crown of thorn',               # 38
#             'dead clam',                    # 39
#         ),
#         # 仅用于可视化的调色板（39 个颜色，长度必须与 classes 一致）
#         palette=[
#             [0, 0, 0],
#             [37, 17, 29],
#             [74, 34, 58],
#             [111, 51, 87],
#             [148, 68, 116],
#             [185, 85, 145],
#             [222, 102, 174],
#             [3, 119, 203],
#             [40, 136, 232],
#             [77, 153, 5],
#             [114, 170, 34],
#             [151, 187, 63],
#             [188, 204, 92],
#             [225, 221, 121],
#             [6, 238, 150],
#             [43, 255, 179],
#             [80, 16, 208],
#             [117, 33, 237],
#             [154, 50, 10],
#             [191, 67, 39],
#             [228, 84, 68],
#             [9, 101, 97],
#             [46, 118, 126],
#             [83, 135, 155],
#             [120, 152, 184],
#             [157, 169, 213],
#             [194, 186, 242],
#             [231, 203, 15],
#             [12, 220, 44],
#             [49, 237, 73],
#             [86, 254, 102],
#             [123, 15, 131],
#             [160, 32, 160],
#             [197, 49, 189],
#             [234, 66, 218],
#             [15, 83, 247],
#             [52, 100, 20],
#             [89, 117, 49],
#             [126, 134, 78],
#         ],
#         # # raw_id -> train_id
#         # # raw 1..39 -> train 0..38 ; raw 0 -> ignore(255)
#         label_map={0: 255, **{i: i - 1 for i in range(1, 40)}}
#     )
#
#     def __init__(self,
#                  img_suffix='_leftImg8bit.png',          # 按你实际数据改成 .png 也行
#                  seg_map_suffix='_gtFine.png',
#                  **kwargs) -> None:
#         super().__init__(img_suffix=img_suffix,
#                          seg_map_suffix=seg_map_suffix,
#                          **kwargs)