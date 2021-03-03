import os
import sys
import csv
import bpy
import time
import glob
import math
import numpy as np
from mathutils import Vector
import argparse
import sys

main_scene = bpy.context.scene
delta = 0
cameraOrigin = None
camera = None
config = None
ring_max = 6

OBJ = None


class setting:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:]

    log_path = 'save.txt'
    path = argv[0]
    phase = argv[1]

    mode = 'test'
    device = 'GPU'
    center = True

    output_name = 'generated_{}/'.format(phase)
    start = 0
    bounding = 30
    rings = [0, 1, 2, 3, 4, 5, 6]

    shader_type = 'default'  # 'gray'
    render_type = 'both'
    use_default_shader = True
    fit = False
    frame_per_ring = 12
    light_type = 'POINT'
    resolution = 512, 512


def reset_blend():
    for scene in bpy.data.scenes:
        for obj in scene.objects:
            scene.objects.unlink(obj)

    for bpy_data_iter in (
            bpy.data.objects,
            bpy.data.meshes,
            bpy.data.lamps,
            bpy.data.cameras,
    ):
        for id_data in bpy_data_iter:
            bpy_data_iter.remove(id_data)


class Scene(object):
    def __init__(self, render_path, frame_end=12, render=False):
        super().__init__()
        # start a new scene
        reset_blend()
        # setup default render path
        bpy.data.scenes[0].render.filepath = render_path
        # setup totals render frames
        self.frame_end = frame_end
        bpy.context.scene.frame_end = self.frame_end

        # setup default params
        self.delta = 2 * math.pi / self.frame_end
        self.cnt = 0
        self.target = None  # main object
        self.camera = None  # main camera
        # setup render mode
        self.is_render = render

    def add_lighting(self, location):
        lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
        lamp_data.energy = 0.5
        lamp_data.use_diffuse = True
        lamp_data.use_specular = False
        # Create new object with our lamp datablock
        lamp_object = bpy.data.objects.new(
            name="New Lamp", object_data=lamp_data)

        # Link lamp object to the scene so it'll appear in this scene
        main_scene.objects.link(lamp_object)

#        Place lamp to a specified location
        lamp_object.location = location

        # And finally select it make active
        lamp_object.select = True
        main_scene.objects.active = lamp_object

    def add_to_scene(self, file_loc, shader_type):
        print(file_loc)
        bpy.ops.import_scene.obj(filepath=file_loc)
        self.target = bpy.context.selected_objects[0]  # <--Fix
        self.target.name = "obj"
        self.target.location = (0, 0, 0)

        if shader_type == 'default':
            pass
        elif shader_type == 'gray':
            mat = bpy.data.materials.new('pixelColor')
            mat.diffuse_color = (0.5, 0.5, 0.5)
            mat.diffuse_shader = 'LAMBERT'
            mat.diffuse_intensity = 1.0
            self.target.data.materials.append(mat)
        else:
            # create the material
            size = 2048, 2048
            # create texture in data block
            tex = bpy.data.images.new(
                "curvature", width=size[0], height=size[1], alpha=True)

            # Create material nodes "CURVATURE"
            ################################################################

            mat = bpy.data.materials.new("CURVATURE")

            # Enable 'Use nodes':
            mat.use_nodes = True
            nt = mat.node_tree
            nodes = nt.nodes
            links = nt.links

            # clear
            while(nodes):
                nodes.remove(nodes[0])

            d_geometry = nodes.new("ShaderNodeNewGeometry")
            d_colorramp = nodes.new("ShaderNodeValToRGB")
            d_emission = nodes.new("ShaderNodeEmission")
            d_output = nodes.new("ShaderNodeOutputMaterial")
            d_image = nodes.new("ShaderNodeTexImage")

            d_geometry.location = (255, 255)
            d_colorramp.location = (255, 255)
            d_emission.location = (255, 255)
            d_output.location = (700, -100)
            d_image.location = (300, 300)

            d_image.image = tex
            d_colorramp.color_ramp.elements[0].position = 0.4
            d_colorramp.color_ramp.elements[1].position = 0.9

            links.new(d_output.inputs['Surface'],
                      d_emission.outputs['Emission'])
            links.new(d_emission.inputs['Color'],
                      d_colorramp.outputs['Color'])
            links.new(d_colorramp.inputs['Fac'],
                      d_geometry.outputs['Pointiness'])

            self.target.data.materials.append(mat)

            # or overwrite an existing material slot via index operator
            # self.target.data.materials[0] = mat
        return self.target

    def new_camera(self, camera_location):
        cam = bpy.data.cameras.new("Camera")
        cam.lens = 18

        self.camera = bpy.data.objects.new("Camera", cam)
        self.camera.location = camera_location
        self.camera.rotation_euler = (90, 0, 0)
        main_scene.camera = self.camera

        main_scene.objects.link(self.camera)

        targetobj = bpy.data.objects[self.target.name]
        pointyobj = bpy.data.objects['Camera']
        # make a new tracking constraint
        ttc = pointyobj.constraints.new(type='TRACK_TO')
        ttc.target = targetobj
        ttc.track_axis = 'TRACK_NEGATIVE_Z'
        ttc.up_axis = 'UP_Y'
        bpy.ops.object.select_all(action='DESELECT')

    def normalize(self):
        # calc ratio
        dimensions = list(self.target.dimensions)
        maxDimension = max(dimensions)
        ratio = 1/maxDimension
        # scale object to 1 blender unit
        self.target.matrix_basis *= self.target.matrix_basis.Scale(
            ratio, 4, (1, 0, 0))
        self.target.matrix_basis *= self.target.matrix_basis.Scale(
            ratio, 4, (0, 1, 0))
        self.target.matrix_basis *= self.target.matrix_basis.Scale(
            ratio, 4, (0, 0, 1))

        # set pivot object to center
        global config
        if config.center:
            self.target.select = True
            bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
            self.target.select = False

        # move object to world origin
        self.target.location = Vector((0, 0, 0))

    def clean(self):
        reset_blend()


def rotateCamera(scene):
    global delta
    angle = delta * scene.frame_current
    rotationMatrix = np.array([[math.cos(angle), math.sin(angle), 0],
                               [math.sin(angle), math.cos(angle), 0],
                               [0, 0, 1]])
    camera.location = np.dot(cameraOrigin, rotationMatrix)

    global config
    global OBJ
    if config.fit:
        for area in bpy.context.screen.areas:
            if area.type == 'VIEW_3D':
                for region in area.regions:
                    if region.type == 'WINDOW':
                        override = {'window': bpy.context.window,
                                    'area': area,
                                    'region': region,
                                    'scene': bpy.context.scene}
                        bpy.ops.view3d.object_as_camera(override)

        bpy.ops.view3d.camera_to_view_selected()


def add_node(tree, psize, pmin, pmax, nodename, out_path):
    links = tree.links
    # create input render layer node
    rl = tree.nodes.new('CompositorNodeRLayers')

    map = tree.nodes.new(type="CompositorNodeMapValue")
    # Size is chosen kind of arbitrarily, try out until you're satisfied with resulting depth map.
    map.size = [psize]
    map.use_min = True
    map.min = [pmin]
    map.use_max = True
    map.max = [pmax]
    links.new(rl.outputs[2], map.inputs[0])

    invert = tree.nodes.new(type="CompositorNodeInvert")
    links.new(map.outputs[0], invert.inputs[1])

    # The viewer can come in handy for inspecting the results in the GUI
    depthViewer = tree.nodes.new(type="CompositorNodeViewer")
    links.new(invert.outputs[0], depthViewer.inputs[0])
    # Use alpha from input.
    links.new(rl.outputs[1], depthViewer.inputs[1])

    fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
    fileOutput.base_path = out_path
    links.new(invert.outputs[0], fileOutput.inputs[0])


def setRotate():
    print("start")
    bpy.app.handlers.frame_change_pre.clear()
    bpy.context.scene.frame_current = 1
    bpy.app.handlers.frame_change_pre.append(rotateCamera)
    bpy.ops.render.render(animation=True)


def take_a_snap(obj_path, output_path, camera_location=(0, -1, 0)):

    global config
    # check blender version == 2.79
    print(bpy.app.version_string)
    # assign a new scene
    scene = Scene(render_path=output_path + '/render/Image####.png',
                  frame_end=config.frame_per_ring)
    # add object and get it
    global OBJ
    OBJ = scene.add_to_scene(obj_path, config.shader_type)
    # normalize it by scale to one unit ( by max edge ) and move the center to origin
    scene.normalize()
    # setup light
    if config.light_type == 'POINT':
        for z in [-1, 1]:
            for x in range(2):
                for y in range(2):
                    location = (pow(-1, x), pow(-1, y), z)
                    scene.add_lighting(location)

    elif config.light_type == 'ENV':
        bpy.data.scenes['Scene'].render.engine = 'CYCLES'
        world = bpy.data.worlds['World']
        world.use_nodes = True

        # changing these values does affect the render.
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value[:3] = (1, 1, 1)
        bg.inputs[1].default_value = 1.0
    else:
        print("BUG LIGHTTTTTT")

    # setup camera
    scene.new_camera(camera_location)

    global delta
    delta = scene.delta
    global camera
    global cameraOrigin
    camera = bpy.data.objects['Camera']
    cameraOrigin = np.array(camera.location)
    # setup render setting
    ##################################################
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree

    # clear default nodes
    if config.render_type != 'render_only':
        for n in tree.nodes:
            tree.nodes.remove(n)
    # depth node
    if config.render_type == 'depth' or config.render_type == 'both':
        add_node(tree, 0.7, 0, 1, "CompositorNodeRLayers",
                 os.path.join(output_path, 'depth/'))
    # mask mode
    if config.render_type == 'mask' or config.render_type == 'both':
        add_node(tree, 0.00001, 0, 255, "CompositorNodeRLayers",
                 os.path.join(output_path, 'mask/'))

    # deselect all objects
    for obj in bpy.context.scene.objects:
        obj.select = False
    # select the target object
    OBJ.select = True

    ##################################################
    # rotate and capture
    bpy.context.scene.frame_current = 1
    setRotate()


class ShrecDataset(object):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.root = config.path
        self.output_path = os.path.join(config.path,
                                        config.output_name)

        fn = os.path.join(config.path,
                          'list/{}.csv'.format(config.phase))
        with open(fn, 'r') as f:
            df = csv.DictReader(f)
            self.list = [config.phase + '/' + x['obj_id'] + '.obj'
                         for x in df]

    def __getitem__(self, index):
        obj = self.list[index].split('\n')[0]
        return (obj)

    def __len__(self):
        return len(self.list)

    def log(self, index):
        f = open(self.cfg.log_path, "w+")
        f.write("%d" % (index))
        f.close()

    def export_2d(self):
        start = self.cfg.start
        print(self.cfg.log_path)
        try:
            f = open(self.cfg.log_path, "r")
            start = f.read()
            if start == '':
                start = self.cfg.start
            else:
                start = int(start)
        except:
            os.mknod(self.cfg.log_path)
            pass

        flag = [None] * 40
        end = len(self) if self.cfg.bounding == - \
            1 else start + self.cfg.bounding - 1
        for id in range(start, end):
            obj_path = os.path.join(self.root, self[id])
            output_path = self[id].split('/')[-1].split('.')[0]

            for k in range(-3, 4):
                z = math.sin(math.pi/6*k) * 1  # 1 Unit
                y = math.cos(math.pi/6*k) * 1  # 1 Unit
                out = os.path.join(
                    self.output_path, 'ring' + str(k+3), output_path)
                take_a_snap(obj_path, out, (0, y, z))
            self.log(id)


if __name__ == "__main__":
    config = setting()

    bpy.context.scene.cycles.device = config.device
    # Get the scene
    scene = bpy.data.scenes["Scene"]
    # Set render resolution
    scene.render.resolution_x, scene.render.resolution_y = config.resolution
    scene.render.resolution_percentage = 100
    # setup dataset
    dataset = ShrecDataset(config)
    dataset.export_2d()
