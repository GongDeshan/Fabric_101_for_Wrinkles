import uipc
import polyscope as ps
from pyuipc.core import Scene, SceneIO

class GUIInfo:
    @staticmethod
    def enabled()->bool:
        level = uipc.Logger.get_level()
        uipc.Logger.set_level(uipc.Logger.Level.Warn)
        config = uipc.Engine.default_config()
        uipc.Logger.set_level(level)
        return config['extras']['gui']['enable']

class SceneGUI:
    def __init__(self, scene:Scene):
        self.scene = scene
        self.scene_io = SceneIO(scene)
        self.ps_trimesh:ps.SurfaceMesh = None
        self.ps_linemesh:ps.CurveNetwork = None
        self.ps_pointcloud:ps.PointCloud = None
        self.ground_obj:uipc.core.Object = None
    
    def register_ground(self, obj: uipc.core.Object)->None:
        ground = obj
        if ground is None: return
        ids = ground.geometries().ids()
        if len(ids) != 1: return
        g, _ = self.scene.geometries().find(ids[0])
        geometry = g.geometry()
        if geometry.instances().size() != 1: return
        P = geometry.instances().find('P')
        N = geometry.instances().find('N')
        if P is None or N is None: return
        p = P.view()[0]
        n = N.view()[0]
        if n[0]!= 0.0 or n[2]!= 0.0: return
        ps.set_ground_plane_height(p[1])
        ps.set_ground_plane_height_mode('manual')
        self.ground_obj = obj
    
    def register(self)->tuple[ps.SurfaceMesh, ps.CurveNetwork, ps.PointCloud]:
        trimesh = self.scene_io.simplicial_surface(2)
        linemesh = self.scene_io.simplicial_surface(1)
        pointcloud = self.scene_io.simplicial_surface(0)
        if(trimesh.vertices().size() != 0):
            self.ps_trimesh = ps.register_surface_mesh('trimesh', trimesh.positions().view().reshape(-1,3), trimesh.triangles().topo().view().reshape(-1,3))
        if(linemesh.vertices().size() != 0):
            self.ps_linemesh = ps.register_curve_network('linemesh', linemesh.positions().view().reshape(-1,3), linemesh.edges().topo().view().reshape(-1,2))
        if(pointcloud.vertices().size() != 0):
            self.ps_pointcloud = ps.register_point_cloud('pointcloud', pointcloud.positions().view().reshape(-1,3))
            thickness = pointcloud.vertices().find(uipc.builtin.thickness).view()
            self.ps_pointcloud.add_scalar_quantity('thickness', thickness)
            self.ps_pointcloud.set_point_radius_quantity('thickness', False)
        return self.ps_trimesh, self.ps_linemesh, self.ps_pointcloud
    
    def update(self):
        if self.ps_trimesh is not None:
            trimesh = self.scene_io.simplicial_surface(2)
            self.ps_trimesh.update_vertex_positions(trimesh.positions().view().reshape(-1,3))
        if self.ps_linemesh is not None:
            linemesh = self.scene_io.simplicial_surface(1)
            self.ps_linemesh.update_node_positions(linemesh.positions().view().reshape(-1,3))
        if self.ps_pointcloud is not None:
            pointcloud = self.scene_io.simplicial_surface(0)
            self.ps_pointcloud.update_point_positions(pointcloud.positions().view().reshape(-1,3))