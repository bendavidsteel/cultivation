import datetime
import json
import logging
import os
import traceback
import math

import librosa
import moderngl_window as mglw
import moderngl
import numpy as np
import soundcard as sc
import glm

import layers
import shape3d
import utils

class LayerManager:
    logger: logging.Logger

    def __init__(self, ctx, config_path, logger):
        self.config_path = config_path
        self.logger = logger

        # Initial shader load
        self.event_handler, self.observer = utils.setup_observer(config_path, self, "load_config")

        self.ctx = ctx
        self.outputs = {}  # Config for each output
        self.layers = {}   # Actual layers for each output
        self.textures = {}  # Textures for each output
        self.framebuffers = {}  # Framebuffers for each output
        self.update_layers = False

        self.premade_shader_map = {
            'solid': layers.SolidColor,
            'osc': layers.Osc,
            'triangle': layers.Triangle,
            'pixelate': layers.Pixelate,
            'scale': layers.Scale,
            'rotate': layers.Rotate,
            'repeat': layers.Repeat,
            'kaleid': layers.Kaleid,
            'shift': layers.Shift,
            'invert': layers.Invert,
            'luma': layers.Luma,
            'gradient': layers.Gradient,
            'noise': layers.Noise,
            'add': layers.Add,
            'multiply': layers.Multiply,
            'modulate': layers.Modulate,
            # Add the new 3D shapes
            'shape3d': shape3d.Shape3D,
            'cube': shape3d.Cube,
            'sphere': shape3d.Sphere,
            'cylinder': shape3d.Cylinder,
            'plane': shape3d.Plane,
            'torus': shape3d.Torus,
        }
        self.python_obj_map = {

        }
        self.load_config()

    def load_config(self):
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)

            for output_name, layers in config.items():
                # ensure layers are correctly formatted
                self.outputs[output_name] = []
                for layer_config in layers:
                    if isinstance(layer_config, str):
                        layer_info = {
                            'name': layer_config
                        }
                    elif isinstance(layer_config, dict):
                        assert 'name' in layer_config, f"{layer_config} must have 'name' attribute"
                        layer_kwargs = {k: v for k, v in layer_config.items() if k != 'name'}
                        
                        # Allow nested objects for 3D shapes
                        for k, val in layer_kwargs.items():
                            val_type = type(val)
                            if val_type not in [int, float, str, list, dict, bool]:
                                raise ValueError(f"Layer kwarg values must be int, float, str, list, bool, or dict, found {val_type}")
                        
                        layer_info = layer_config.copy()
                    else:
                        raise ValueError(f"Layer must be either a str or a dict, got {layer_config}")
                    
                    self.outputs[output_name].append(layer_info)

            self.update_layers = True
        except Exception as e:
            self.logger.error(f"Error loading config file: {e}, {traceback.format_exc()}")

    def update(self):
        try:
            if self.update_layers:
                # Clean up existing resources
                for output_name, output_layers in self.layers.items():
                    for layer in output_layers:
                        layer.exit()
                    
                    if output_name in self.textures:
                        for texture in self.textures[output_name]:
                            texture.release()
                    
                    if output_name in self.framebuffers:
                        for framebuffer in self.framebuffers[output_name]:
                            framebuffer.release()
                
                # Initialize new collections
                self.layers = {}
                self.textures = {}
                self.framebuffers = {}
                
                # Create layers for each output
                for output_name, layers_info in self.outputs.items():
                    self.layers[output_name] = []
                    self.textures[output_name] = []
                    self.framebuffers[output_name] = []
                    
                    for layer_info in layers_info:
                        layer_name = layer_info['name']
                        layer_kwargs = {k: v for k, v in layer_info.items() if k != 'name'}

                        try:
                            # check to see what kind of layer this is
                            if layer_name == 'primitives':
                                new_layer = shape3d.Primitives(self.ctx, self.logger, **layer_kwargs)
                            elif layer_name in self.premade_shader_map:
                                new_layer = self.premade_shader_map[layer_name](self.ctx, self.logger, **layer_kwargs)
                            elif layer_name in self.python_obj_map:
                                new_layer = layers.PythonLayer(self.ctx, self.logger, self.python_obj_map[layer_name], **layer_kwargs)
                            else:
                                new_layer = layers.CustomShader(self.ctx, self.logger, layer_name, **layer_kwargs)

                            if len(self.layers[output_name]) == 0:
                                assert new_layer.source_layer, f"First layer {new_layer} must be a source layer"

                            self.layers[output_name].append(new_layer)
                            texture = self.ctx.texture((self.ctx.screen.width, self.ctx.screen.height), 4)
                            self.textures[output_name].append(texture)
                            framebuffer = self.ctx.framebuffer(color_attachments=[texture])
                            self.framebuffers[output_name].append(framebuffer)
                        except Exception as e:
                            self.logger.error(f"Error creating layer {layer_name}: {e}")
                    
                self.update_layers = False

            # Update all layers
            for output_name, output_layers in self.layers.items():
                for layer in output_layers:
                    layer.update()
                        
        except Exception as e:
            self.logger.error(f"Error loading layers at {datetime.datetime.now()}: {e}, {traceback.format_exc()}")
            self.successful_load = False
        else:
            self.successful_load = True

    def exit(self):
        for layer in self.layers:
            layer.exit()

        for texture in self.textures:
            texture.release()

        for framebuffer in self.framebuffers:
            framebuffer.release()

    def render(self, resolution, time, fft):
        # Set up proper 3D rendering state
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)
        
        # Dictionary to track the final texture of each output for this frame
        final_textures = {}
        
        # Process each output
        for output_name, output_layers in self.layers.items():
            if not output_layers:
                continue
            
            output_textures = self.textures[output_name]
            output_framebuffers = self.framebuffers[output_name]
            
            for i, (layer, framebuffer) in enumerate(zip(output_layers, output_framebuffers)):
                # Render to the framebuffer
                framebuffer.use()
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)
                
                kwargs = {}
                
                # Use the previous texture in this output's chain as input if available
                if hasattr(layer, 'get_uniforms') and 'input_texture' in layer.get_uniforms() and i > 0:
                    output_textures[i-1].use(location=0)
                    kwargs['input_texture'] = 0
                
                # Make all outputs' final textures available as uniforms
                texture_location = 1
                for other_output in self.layers.keys():
                    # Skip if output doesn't exist yet
                    if other_output not in self.textures or not self.textures[other_output]:
                        continue
                    
                    # Use the final texture from the previous frame
                    other_final_texture = self.textures[other_output][-1]
                    other_final_texture.use(location=texture_location)
                    kwargs[other_output] = texture_location
                    texture_location += 1
                
                # Add standard parameters
                kwargs['fft'] = fft
                kwargs['resolution'] = resolution
                kwargs['time'] = time
                
                try:
                    layer.render(**kwargs)
                except Exception as e:
                    self.logger.error(f"Error rendering layer {layer} for output {output_name}: {e}")
                    # render last layer if error occurs
                    output_textures[i-1].use(location=0)
                    final_layer = layers.Identity(self.ctx, self.logger)
                    kwargs = {'input_texture': 0, 'resolution': resolution}
                    final_layer.render(**kwargs)
                    
            # Store the final texture for this output
            if output_textures:
                final_textures[output_name] = output_textures[-1]
        
        # Render o0 to the screen if it exists
        if 'o0' in final_textures:
            self.ctx.screen.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            final_textures['o0'].use(location=0)
            final_layer = layers.Identity(self.ctx, self.logger)
            kwargs = {'input_texture': 0, 'resolution': resolution}
            final_layer.render(**kwargs)
        
        # Reset 3D state
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.CULL_FACE)

    def get_smoothing_factor(self):
        return 1


class RealTimeShaderApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Real-Time Shader Loader"
    window_size = (1280, 720)
    resource_dir = os.path.normpath(os.path.join(os.path.dirname(__file__)))
    logger = logging.getLogger(__name__)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.setLevel(logging.DEBUG)

        config_path = './config.json'

        self.layer_manager = LayerManager(self.ctx, config_path, self.logger)

        samplerate = 44100  # Sample rate in Hz
        self.duration = 5  # Duration of each recording in seconds
        self.fft_size = 4  # Number of FFT points
        self.fft_memory = 8192  # Number of FFT points to average over
        self.smoothing_factor = 0.0001

        # Get the default microphone
        microphone = sc.default_microphone()
        self.mic = microphone.recorder(samplerate=samplerate, channels=1, blocksize=self.fft_size).__enter__()
        self.fft = np.zeros((self.fft_size-1, self.fft_memory))

    def on_render(self, time, frame_time):
        resolution = (self.window_size[0], self.window_size[1])
        audio_data = self.mic.record()
        # Convert to mono by averaging channels if needed
        audio_data = np.mean(audio_data, axis=1)
        # Perform FFT using librosa
        fft_data = librosa.stft(audio_data, n_fft=self.fft_size)
        # smooth the fft data
        fft = np.abs(fft_data)
        fft /= np.sum(fft)
        self.fft[:, fft.shape[1]:] = self.fft[:, :-fft.shape[1]]
        self.fft[:, :fft.shape[1]] = fft
        moving_avg_weights = (np.ones(self.fft_memory) - self.smoothing_factor) ** np.arange(self.fft_memory)
        fft = np.sum(self.fft * moving_avg_weights, axis=1)
        self.layer_manager.update()
        self.smoothing_factor = self.layer_manager.get_smoothing_factor()
        self.layer_manager.render(resolution, time, fft)

    def __del__(self):
        self.mic.__exit__(None, None, None)
        self.layer_manager.exit()
        self.observer.stop()
        self.observer.join()

if __name__ == '__main__':
    RealTimeShaderApp.run()