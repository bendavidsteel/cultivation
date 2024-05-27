import os

import numpy as np
from watchdog.observers import Observer

from layer import BaseLayer
import utils

class BaseLayer:
    def __init__():
        raise NotImplementedError("BaseLayer is an abstract class and cannot be instantiated")

    def update(self):
        pass

    def render(self, resolution, time):
        raise NotImplementedError("BaseLayer is an abstract class and cannot be instantiated")
    
    def exit(self):
        pass

class BaseShader(BaseLayer):
    def __init__(self, ctx, **kwargs):
        self.ctx = ctx
        self.kwargs = kwargs
        self.load_shader()

    def load_shader(self):
        self.program = self.ctx.program(
            vertex_shader=self.get_vertex_shader(),
            fragment_shader=self.get_fragment_shader()
        )
        self.load_vao()

    def load_vao(self):
        vertices = np.array([
            -1.0, -1.0,
            1.0, -1.0,
            1.0, 1.0,
            -1.0, -1.0,
            1.0, 1.0,
            -1.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)

        # We control the 'in_vert' and `in_color' variables
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_vert to the first 2 floats
                self.vbo.bind('in_vert', layout='2f'),
            ],
        )

    def render(self, resolution, time):
        if 'resolution' in self.program:
            self.program['resolution'] = resolution
        if 'time' in self.program:
            self.program['time'] = time
        for k, val in self.kwargs.items():
            if k in self.program:
                self.program[k] = val
        self.vao.render()

    def get_vertex_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        in vec3 in_vert;

        void main() {
            gl_Position = vec4(in_vert, 1.0);
        }
        """
    
    def get_fragment_shader(self):
        raise NotImplementedError("PremadeShader is an abstract class and cannot be instantiated")
    
class CustomShader(BaseShader):
    def __init__(self, ctx, shader_name):
        self.shader_name = shader_name
        self.reload_shaders = False
        self.ctx = ctx
        self.read_shader_files()
        self.load_shader()
        self.vert_shader_event_handler, self.vert_shader_observer = utils.setup_observer(self.vert_shader_file_path, self, "vertex_file_changed")
        self.frag_shader_event_handler, self.frag_shader_observer = utils.setup_observer(self.frag_shader_file_path, self, "fragment_file_changed")

    def vertex_file_changed(self):
        self.reload_shaders = True
        self.load_vertex_file()

    def fragment_file_changed(self):
        self.reload_shaders = True
        self.load_fragment_file()

    def load_vertex_file(self):
        with open(self.vert_shader_file_path, 'r') as f:
            self.vertex_shader_code = f.read()

    def load_fragment_file(self):
        with open(self.frag_shader_file_path, 'r') as f:
            self.fragment_shader_code = f.read()

    def read_shader_files(self):
        shader_dir_path = os.path.join(os.path.dirname(__file__), 'shaders')
        shader_files = os.listdir(shader_dir_path)
        
        frag_shader_file_name = f"{self.shader_name}.frag"
        self.frag_shader_file_path = os.path.join(shader_dir_path, frag_shader_file_name)
        self.load_fragment_file()

        vert_shader_file_name = f"{self.shader_name}.vert"
        assert frag_shader_file_name in shader_files, f"Fragment shader file {frag_shader_file_name} not found"
        if vert_shader_file_name in shader_files:
            self.vert_shader_file_path = os.path.join(shader_dir_path, vert_shader_file_name)
            print(f"Vertex shader file {vert_shader_file_name} not found, using generic vert shader")
            self.load_vertex_file()
        else:
            self.vertex_shader_code = self.get_vertex_shader()

    def get_vertex_shader(self):
        return self.vertex_shader_code
    
    def get_fragment_shader(self):
        return self.fragment_shader_code
    
    def update(self):
        if self.reload_shaders:
            self.load_shader()

    def exit(self):
        self.vert_shader_observer.stop()
        self.vert_shader_observer.join()
        self.frag_shader_observer.stop()
        self.frag_shader_observer.join()


class PythonLayer(BaseShader):
    def __init__(self, ctx, obj, **kwargs):
        assert hasattr(obj, "update"), "obj must have update method"
        assert hasattr(obj, "render"), "obj must have render method"
        self.obj = obj
        self.kwargs = kwargs
        super().__init__(ctx)
    
    def get_fragment_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        uniform sampler2DRect tex;

        out vec4 fragColor;
        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;
            vec3 rgb = texture(tex, uv);
            fragColor = vec4(uv, 1.0); // Red color
        }
        """

    def update(self):
        self.obj.update(**self.kwargs)

    def render(self, resolution, time):
        display = self.obj.render()

        # TODO load display into an fbo, and add as a texture on the shader

        if 'resolution' in self.program:
            self.program['resolution'] = resolution
        if 'time' in self.program:
            self.program['time'] = time
        self.vao.render()
        

class SolidColor(BaseShader):
    def get_fragment_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        out vec4 fragColor;
        void main() {
            fragColor = vec4(1.0, 0.0, 0.0, 1.0); // Red color
        }
        """
    
class Osc(BaseShader):
    def get_fragment_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        out vec4 fragColor;
        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;
            float b = sin(uv.x * 10.0 + time) * 0.5 + 0.5;
            fragColor = vec4(b, b, b, 1.0); // Red color
        }        
        """
    
class Triangle(BaseShader):
    def get_fragment_shader(self):
        return '''
            #version 330

            in vec3 v_color;
            out vec4 f_color;

            void main() {
                // We're not interested in changing the alpha value
                f_color = vec4(v_color, 1.0);
                f_color.rgb = pow(f_color.rgb, vec3(1.0 / 2.2));
            }
        '''
    
    def load_vao(self):
        # Point coordinates are put followed by the vec3 color values
        vertices = np.array([
            # x, y, red, green, blue
            0.0, 0.8, 1.0, 0.0, 0.0,
            -0.6, -0.8, 0.0, 1.0, 0.0,
            0.6, -0.8, 0.0, 0.0, 1.0,
        ], dtype='f4')

        self.vbo = self.ctx.buffer(vertices)

        # We control the 'in_vert' and `in_color' variables
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                # Map in_vert to the first 2 floats
                # Map in_color to the next 3 floats
                self.vbo.bind('in_vert', 'in_color', layout='2f 3f'),
            ],
        )
