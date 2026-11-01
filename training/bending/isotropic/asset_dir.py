import datetime
from pathlib import Path
class AssetDir:
    this_file=Path(__file__).resolve().parent
    @staticmethod
    def asset_path():return str(AssetDir.this_file/'assets')
    @staticmethod
    def tetmesh_path():return str(AssetDir.this_file/'assets/sim_data/tetmesh')
    @staticmethod
    def trimesh_path():return str(AssetDir.this_file/'assets/sim_data/trimesh')
    @staticmethod
    def folder(file):return Path(file).resolve().parent
    @staticmethod
    def output_path(file):
        p=AssetDir.this_file/'logs'/(Path(file).stem+'_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f'))/'backend'
        p.mkdir(parents=True,exist_ok=False)
        return str(p)
