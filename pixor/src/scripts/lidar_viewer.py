import os
import numpy as np
from OpenGL.GL import glLineWidth
import pyqtgraph as pg
import pyqtgraph.opengl as gl


class Object3d(object):
    def __init__(self, obj_cls, hwl, location, rz):

        # extract label, truncation, occlusion
        self.type = obj_cls # 'Car', 'Pedestrian', ...
        self.hwl = hwl # box height
        self.location = location # location (x,y,z) in lidar coord.
        self.rz = rz # yaw angle (around Z-axis in lidar coordinates) [-pi..pi]

    def print_object(self):
        print('Type : ', self.type)
        print('HWL : ', self.hwl)
        print('Location: ', self.location)
        print('Rz: ', self.rz)


def create_bbox_mesh(p3d, gt_boxes3d):
    b = gt_boxes3d
    for k in range(0,4):
        i,j=k,(k+1)%4
        p3d.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]])
        i,j=k+4,(k+1)%4 + 4
        p3d.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]])
        i,j=k,k+4
        p3d.add_line([b[i,0],b[i,1],b[i,2]], [b[j,0],b[j,1],b[j,2]])


class plot3d(object):
    def __init__(self):
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        glLineWidth(3)
        coord.setSize(3,3,3)
        self.view.addItem(coord)
    def add_points(self, points, colors):
        points_item = gl.GLScatterPlotItem(pos=points, size=2, color=colors)
        self.view.addItem(points_item)
    def add_line(self, p1, p2):
        lines = np.array([[p1[0], p1[1], p1[2]],
                          [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=(1,0,0,1), width=3, antialias=True)
        self.view.addItem(lines_item)
    def show(self):
        self.view.show()
        self.app.exec()


def show_lidar_with_boxes(pc_velo, objects, calib):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    pc_inte = pc_velo[:, 3]
    pc_color = inte_to_rgb(pc_inte)
    p3d.add_points(points, pc_color)
    for obj in objects:
        if obj.type=='DontCare':continue
        # Draw 3d bounding box
        box3d_pts_2d, box3d_pts_3d = compute_box_3d(obj, calib.P) 
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        create_bbox_mesh(p3d, box3d_pts_3d_velo)
        # Draw heading arrow
        ori3d_pts_2d, ori3d_pts_3d = compute_orientation_3d(obj, calib.P)
        ori3d_pts_3d_velo = calib.project_rect_to_velo(ori3d_pts_3d)
        x1,y1,z1 = ori3d_pts_3d_velo[0,:]
        x2,y2,z2 = ori3d_pts_3d_velo[1,:]
        p3d.add_line([x1,y1,z1], [x2,y2,z2])
    p3d.show()


def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte-minimum) / (maximum - minimum)
    b = (np.maximum((1 - ratio), 0))
    r = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()

# -----------------------------------------------------------------------------------------

if __name__ == '__main__':
    dataset = kitti_object('./', 'training')
    data_idx = 5
    # PC
    lidar_data = dataset.get_lidar(data_idx)
    print(lidar_data.shape)
    # OBJECTS
    objects = dataset.get_label_objects(data_idx)
    objects[0].print_object()
    # CALIB
    calib = dataset.get_calibration(data_idx)
    print(calib.P)
    # Show
    show_lidar_with_boxes(lidar_data, objects, calib)