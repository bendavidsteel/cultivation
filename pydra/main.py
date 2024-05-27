import datetime
import json
import os
import re

import numpy as np
import moderngl
import moderngl_window as mglw


import layers
import utils

class LayerManager:
    def __init__(self, ctx, config_path):
        self.config_path = config_path

        # Initial shader load
        self.event_handler, self.observer = utils.setup_observer(config_path, self, "load_config")

        self.ctx = ctx
        self.layers = []
        self.textures = []
        self.framebuffers = []
        self.update_layers = False

        self.premade_shader_map = {
            'solid': layers.SolidColor,
            'osc': layers.Osc,
            'triangle': layers.Triangle,
        }
        self.python_obj_map = {

        }
        self.load_config()


    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                layer_info = json.load(f)

            if "layers" not in layer_info:
                raise ValueError("No 'layers' key in layer_info")
            
            self.layers_config = layer_info['layers']

            # ensure layers are correctly formatted
            self.layers_info = []
            for layer_config in self.layers_config:
                if isinstance(layer_config, str):
                    layer_info = {
                        'name': layer_config
                    }
                elif isinstance(layer_config, dict):
                    assert 'name' in layer_config, f"{layer_config} must have 'name' attribute"
                    layer_kwargs = {k: v for k, v in layer_config if k != 'name'}
                    for k, val in layer_kwargs:
                        val_type = type(val)
                        if val_type is str:
                            assert val in self.layer_inputs, f"Layer kwarg values must be in preexisting input list, found {val}"
                        elif val_type not in [int, float]
                            raise ValueError("")
                    layer_info = layer_config.copy()
                else:
                    raise ValueError(f"Layer must be either a str or a dict, got {layer_config}")
                
                self.layers_info.append(layer_info)

            self.update_layers = True
        except Exception as e:
            print(f"Error loading config file: {e}")

    def update(self):
        try:
            if self.update_layers:
                for layer in self.layers:
                    layer.exit()

                self.layers = []
                for layer_info in self.layers_info:
                    layer_name = layer_info['name']
                    layer_kwargs = {k: v for k, v in layer_info.items() if k != 'name'}

                    # check to see what kind of layer this is
                    if layer_name in self.premade_shader_map:
                        new_layer = self.premade_shader_map[layer_name](self.ctx, **layer_kwargs)
                    elif layer_name in self.python_obj_map:
                        new_layer = layers.PythonLayer(self.ctx, self.python_obj_map[layer_name], **layer_kwargs)
                    else:
                        new_layer = layers.CustomShader(self.ctx, layer_name, **layer_kwargs)
                    self.layers.append(new_layer)
                    
        except Exception as e:
            print(f"Error loading layers at {datetime.datetime.now()}: {e}")
            self.successful_load = False
        else:
            self.successful_load = True

    def exit(self):
        for layer in self.layers:
            layer.exit()
        for tex in self.textures:
            tex.release()
        for fb in self.framebuffers:
            fb.release()

    def render(self, resolution, time):
        if self.vao and self.successful_load:
            try:
                if 'resolution' in self.prog:
                    self.prog['resolution'] = resolution
                if 'time' in self.prog:
                    self.prog['time'] = time
                self.vao.render()
            except Exception as e:
                print(f"Error rendering: {e}")


class RealTimeShaderApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Real-Time Shader Loader"
    window_size = (1280, 720)
    resource_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), 'layers'))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        dir_path = os.path.dirname(__file__)
        config_path = os.path.join(dir_path, 'config.json')

        self.layer_manager = LayerManager(self.ctx, config_path)

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        resolution = (self.window_size[0], self.window_size[1])
        self.layer_manager.update()
        self.layer_manager.render(resolution, time)

    def __del__(self):
        self.layer_manager.exit()
        self.observer.stop()
        self.observer.join()

if __name__ == '__main__':
    RealTimeShaderApp.run()