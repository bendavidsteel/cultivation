import datetime
import json
import os
import traceback

import librosa
import moderngl_window as mglw
import numpy as np
import soundcard as sc

import layers
import utils

class LayerManager:
    def __init__(self, ctx, config_path):
        self.config_path = config_path

        # Initial shader load
        self.event_handler, self.observer = utils.setup_observer(config_path, self, "load_config")

        self.ctx = ctx
        self.layers = []
        self.update_layers = False

        self.texture_a = self.ctx.texture((self.ctx.screen.width, self.ctx.screen.height), 4)
        self.texture_b = self.ctx.texture((self.ctx.screen.width, self.ctx.screen.height), 4)

        self.framebuffer_a = self.ctx.framebuffer(color_attachments=[self.texture_a])
        self.framebuffer_b = self.ctx.framebuffer(color_attachments=[self.texture_b])

        self.texture_out = self.ctx.texture((self.ctx.screen.width, self.ctx.screen.height), 4)
        self.framebuffer_out = self.ctx.framebuffer(color_attachments=[self.texture_out])

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
                    layer_kwargs = {k: v for k, v in layer_config.items() if k != 'name'}
                    for k, val in layer_kwargs.items():
                        val_type = type(val)
                        if val_type not in [int, float, str]:
                            raise ValueError(f"Layer kwarg values must be int, float, or str, found {val_type}")
                    layer_info = layer_config.copy()
                else:
                    raise ValueError(f"Layer must be either a str or a dict, got {layer_config}")
                
                self.layers_info.append(layer_info)

            self.update_layers = True
        except Exception as e:
            print(f"Error loading config file: {e}, {traceback.format_exc()}")

    def update(self):
        try:
            if self.update_layers:
                for layer in self.layers:
                    layer.exit()

                self.layers = []
                for layer_info in self.layers_info:
                    layer_name = layer_info['name']
                    layer_kwargs = {k: v for k, v in layer_info.items() if k != 'name'}

                    try:
                        # check to see what kind of layer this is
                        if layer_name in self.premade_shader_map:
                            new_layer = self.premade_shader_map[layer_name](self.ctx, **layer_kwargs)
                        elif layer_name in self.python_obj_map:
                            new_layer = layers.PythonLayer(self.ctx, self.python_obj_map[layer_name], **layer_kwargs)
                        else:
                            new_layer = layers.CustomShader(self.ctx, layer_name, **layer_kwargs)

                        if len(self.layers) == 0:
                            assert new_layer.source_layer, f"First layer {new_layer} must be a source layer"

                        self.layers.append(new_layer)
                    except Exception as e:
                        print(f"Error creating layer {layer_name}: {e}")
                self.update_layers = False

            for layer in self.layers:
                layer.update()
                    
        except Exception as e:
            print(f"Error loading layers at {datetime.datetime.now()}: {e}, {traceback.format_exc()}")
            self.successful_load = False
        else:
            self.successful_load = True

    def exit(self):
        for layer in self.layers:
            layer.exit()

        self.texture_a.release()
        self.texture_b.release()
        self.texture_out.release()
        self.framebuffer_a.release()
        self.framebuffer_b.release()
        self.framebuffer_out.release()

    def render(self, resolution, time, fft):
        for i, layer in enumerate(self.layers[:-1]):
            # First shader renders to the first framebuffer
            if i % 2 == 0:
                self.framebuffer_a.use()
            else:
                self.framebuffer_b.use()
            self.ctx.clear(0.0, 0.0, 0.0, 1.0)
            kwargs = {}
            if i != 0:
                # Subsequent shaders use the previous texture as input
                if (i-1) % 2 == 0:
                    self.texture_b.use(location=0)
                else:
                    self.texture_a.use(location=0)
                kwargs['input_texture'] = 0 # Texture unit 0

            kwargs['fft'] = fft
            kwargs['resolution'] = resolution
            kwargs['time'] = time
            layer.render(**kwargs)
        
        if len(self.layers) > 0:
            # Final output to the screen
            self.ctx.screen.use()
            kwargs = {}
            if len(self.layers) > 1:
                if len(self.layers) % 2 == 0:
                    self.texture_a.use(location=0)
                else:
                    self.texture_b.use(location=0)
                kwargs['input_texture'] = 0

            kwargs['fft'] = fft
            kwargs['resolution'] = resolution
            kwargs['time'] = time
            self.layers[-1].render(**kwargs)


class RealTimeShaderApp(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "Real-Time Shader Loader"
    window_size = (1280, 720)
    resource_dir = os.path.normpath(os.path.join(os.path.dirname(__file__)))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        config_path = './config.json'

        self.layer_manager = LayerManager(self.ctx, config_path)

        samplerate = 44100  # Sample rate in Hz
        self.duration = 1  # Duration of each recording in seconds
        self.fft_size = 4  # Number of FFT points

        # Get the default microphone
        microphone = sc.default_microphone()
        self.mic = microphone.recorder(samplerate=samplerate, channels=1, blocksize=self.fft_size).__enter__()

    def render(self, time, frame_time):
        self.ctx.clear(1.0, 1.0, 1.0)
        resolution = (self.window_size[0], self.window_size[1])
        audio_data = self.mic.record()
        # Convert to mono by averaging channels if needed
        audio_data = np.mean(audio_data, axis=1)
        # Perform FFT using librosa
        fft_data = librosa.stft(audio_data, n_fft=self.fft_size)[:,-1]
        fft = np.abs(fft_data).flatten() / np.max(np.abs(fft_data))
        self.layer_manager.update()
        self.layer_manager.render(resolution, time, fft)

    def __del__(self):
        self.mic.__exit__(None, None, None)
        self.layer_manager.exit()
        self.observer.stop()
        self.observer.join()

if __name__ == '__main__':
    RealTimeShaderApp.run()