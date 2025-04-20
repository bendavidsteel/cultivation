import logging
import os

import moderngl
import numpy as np
from watchdog.observers import Observer

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
    logger: logging.Logger
    program: moderngl.Program

    def __init__(self, ctx, logger, **kwargs):
        self.ctx = ctx
        self.logger = logger
        self.kwargs = kwargs
        self.source_layer = True
        self.load_shader()

    def get_uniforms(self):
        if self.program:
            return [k for k, v in self.program._members.items() if isinstance(v, moderngl.Uniform)]
        else:
            return []

    def load_shader(self):
        try:
            self.program = self.ctx.program(
                vertex_shader=self.get_vertex_shader(),
                fragment_shader=self.get_fragment_shader()
            )
        except moderngl.Error as e:
            raise moderngl.Error(f"Error in layer: {str(self)}: {e}")
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
            self.program,
            [
                # Map in_vert to the first 2 floats
                self.vbo.bind('in_vert', layout='2f'),
            ],
        )

    def render(self, **kwargs):
        kwargs.update(self.kwargs)
        for k, val in kwargs.items():
            if k in self.program:
                if isinstance(val, str):
                    self.program[k] = utils.eval_statement(val, kwargs, 1.0, k, self.logger)
                else:
                    self.program[k] = val
        self.vao.render()

    def default_fragment_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        out vec4 fragColor;
        void main() {
            fragColor = vec4(0.0, 0.0, 0.0, 1.0); // black color
        }
        """

    def default_vertex_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        in vec3 in_vert;

        void main() {
            gl_Position = vec4(in_vert, 1.0);
        }
        """
    
    def get_vertex_shader(self):
        return self.default_vertex_shader()
    
    def get_fragment_shader(self):
        raise NotImplementedError("PremadeShader is an abstract class and cannot be instantiated")
    
    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.kwargs})"
    
class CustomShader(BaseShader):
    def __init__(self, ctx, logger, shader_name, **kwargs):
        self.shader_name = shader_name
        self.reload_shaders = False
        self.source_layer = True
        self.ctx = ctx
        self.logger = logger
        self.kwargs = kwargs
        self.read_shader_files()
        self.frag_shader_event_handler, self.frag_shader_observer = utils.setup_observer(self.frag_shader_file_path, self, "fragment_file_changed")
        if self.vert_shader_file_path:
            self.vert_shader_event_handler, self.vert_shader_observer = utils.setup_observer(self.vert_shader_file_path, self, "vertex_file_changed")
        
        try:
            self.load_shader()
        except Exception as e:
            print(f"Error loading shader: {e}")
            self.fragment_shader_code = self.default_fragment_shader()
            self.vertex_shader_code = self.default_vertex_shader()
            self.load_shader()

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
        shader_dir_path = './shaders/'
        shader_files = os.listdir(shader_dir_path)
        
        frag_shader_file_name = f"{self.shader_name}.frag"
        self.frag_shader_file_path = os.path.join(shader_dir_path, frag_shader_file_name)
        self.load_fragment_file()

        vert_shader_file_name = f"{self.shader_name}.vert"
        assert frag_shader_file_name in shader_files, f"Fragment shader file {frag_shader_file_name} not found"
        if vert_shader_file_name in shader_files:
            self.vert_shader_file_path = os.path.join(shader_dir_path, vert_shader_file_name)
            self.load_vertex_file()
        else:
            self.vert_shader_file_path = None
            print(f"Vertex shader file {vert_shader_file_name} not found, using generic vert shader")
            self.vertex_shader_code = self.default_vertex_shader()

    def get_vertex_shader(self):
        return self.vertex_shader_code
    
    def get_fragment_shader(self):
        return self.fragment_shader_code
    
    def update(self):
        if self.reload_shaders:
            old_program = self.program
            try:
                self.load_shader()
            except Exception as e:
                self.program = old_program
                self.load_vao()
                print(f"Error reloading shader: {e}")
            else:
                print(f"Succesfully reloaded shader {self.shader_name}")
            self.reload_shaders = False

    def exit(self):
        if self.vert_shader_file_path:
            self.vert_shader_observer.stop()
            self.vert_shader_observer.join()
        self.frag_shader_observer.stop()
        self.frag_shader_observer.join()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.shader_name})"


class PythonLayer(BaseShader):
    def __init__(self, ctx, logger, obj, **kwargs):
        assert hasattr(obj, "update"), "obj must have update method"
        assert hasattr(obj, "render"), "obj must have render method"
        self.obj = obj
        self.kwargs = kwargs
        super().__init__(ctx, logger)
    
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

class TransformShader(BaseShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.source_layer = False

class Pixelate(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'pixelX' not in self.kwargs:
            self.kwargs['pixelX'] = 10
        if 'pixelY' not in self.kwargs:
            self.kwargs['pixelY'] = 10

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform int pixelX;
            uniform int pixelY;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec2 pixel = vec2(pixelX, pixelY) * 1.0 / resolution.xy;

                vec2 coord = uv - mod(uv, pixel);
                vec3 color = texture(input_texture, coord).rgb;

                fragColor = vec4(color, 1.0);
            }
            """
    
class Scale(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'scaleX' not in self.kwargs:
            self.kwargs['scaleX'] = 1.0
        if 'scaleY' not in self.kwargs:
            self.kwargs['scaleY'] = 1.0
        if 'amount' not in self.kwargs:
            self.kwargs['amount'] = 1.5
        if 'offsetX' not in self.kwargs:
            self.kwargs['offsetX'] = 0.5
        if 'offsetY' not in self.kwargs:
            self.kwargs['offsetY'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330
            
            uniform sampler2D input_texture;
            
            uniform float scaleX;
            uniform float scaleY;
            uniform float amount;
            uniform float offsetX;
            uniform float offsetY;
            
            uniform vec2 resolution;
            uniform float time;
            
            out vec4 fragColor;
            
            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec2 center = vec2(offsetX, offsetY);
                vec2 scaledUV = center + (uv - center) * vec2(scaleX, scaleY) * amount;
                vec3 color = texture(input_texture, scaledUV).rgb;
                fragColor = vec4(color, 1.0);
            }
            """
    
class Rotate(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'angle' not in self.kwargs:
            self.kwargs['angle'] = 20
        if 'offsetX' not in self.kwargs:
            self.kwargs['offsetX'] = 0.5
        if 'offsetY' not in self.kwargs:
            self.kwargs['offsetY'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform float angle;
            uniform float offsetX;
            uniform float offsetY;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec2 center = vec2(offsetX, offsetY);
                vec2 rotatedUV = center + (uv - center) * mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
                vec3 color = texture(input_texture, rotatedUV).rgb;
                fragColor = vec4(color, 1.0);
            }
            """
    
class Repeat(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'repeatX' not in self.kwargs:
            self.kwargs['repeatX'] = 2
        if 'repeatY' not in self.kwargs:
            self.kwargs['repeatY'] = 2

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform int repeatX;
            uniform int repeatY;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec2 repeat = vec2(repeatX, repeatY);
                vec2 coord = mod(uv, 1.0 / repeat);
                vec3 color = texture(input_texture, coord).rgb;
                fragColor = vec4(color, 1.0);
            }
            """
    

class Kaleid(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'sides' not in self.kwargs:
            self.kwargs['sides'] = 6
        if 'angle' not in self.kwargs:
            self.kwargs['angle'] = 0
        if 'offsetX' not in self.kwargs:
            self.kwargs['offsetX'] = 0.5
        if 'offsetY' not in self.kwargs:
            self.kwargs['offsetY'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform int sides;
            uniform float angle;
            uniform float offsetX;
            uniform float offsetY;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec2 center = vec2(offsetX, offsetY);
                vec2 dir = uv - center;
                float dist = length(dir);
                float a = atan(dir.y, dir.x) + radians(angle);
                float r = float(sides) * a / (2.0 * 3.14159265359);
                float f = abs(fract(r) - 0.5) * 2.0;
                vec2 coord = center + vec2(cos(a), sin(a)) * dist * f;
                vec3 color = texture(input_texture, coord).rgb;
                fragColor = vec4(color, 1.0);
            }
            """
    
class Shift(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'r' not in self.kwargs:
            self.kwargs['r'] = 0.5
        if 'g' not in self.kwargs:
            self.kwargs['g'] = 0.5
        if 'b' not in self.kwargs:
            self.kwargs['b'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform float r;
            uniform float g;
            uniform float b;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = texture(input_texture, uv).rgb;
                // shift the color
                color.r = fract(color.r + r);
                color.g = fract(color.g + g);
                color.b = fract(color.b + b);
                fragColor = vec4(color, 1.0);
            }
            """
    
class Invert(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'amount' not in self.kwargs:
            self.kwargs['amount'] = 1.0

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform float amount;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = texture(input_texture, uv).rgb;
                // invert the color
                color = mix(color, 1.0 - color, amount);
                fragColor = vec4(color, 1.0);
            }
            """
    

class Luma(TransformShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'amount' not in self.kwargs:
            self.kwargs['amount'] = 1.0

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform float amount;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = texture(input_texture, uv).rgb;
                // convert to luma
                float luma = dot(color, vec3(0.299, 0.587, 0.114));
                color = vec3(luma);
                color = mix(color, vec3(luma), amount);
                fragColor = vec4(color, 1.0);
            }
            """
    

class Gradient(BaseShader):
    def get_fragment_shader(self):
        return """
            #version 330

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                // gradually changing gradient texture
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = vec3(0.5 + 0.5 * cos(time + uv.x), 0.5 + 0.5 * sin(time + uv.y), 0.5 + 0.5 * cos(time + uv.x));
                fragColor = vec4(color, 1.0);
            }
            """

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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # set defaults
        if 'scale' not in self.kwargs:
            self.kwargs['scale'] = 10
        if 'speed' not in self.kwargs:
            self.kwargs['speed'] = 1

    def get_fragment_shader(self):
        return """
        #version 330

        uniform vec2 resolution;
        uniform float time;

        uniform float scale;
        uniform float speed;

        out vec4 fragColor;
        void main() {
            vec2 uv = gl_FragCoord.xy / resolution.xy;
            float b = sin(uv.x * scale + time * speed) * 0.5 + 0.5;
            fragColor = vec4(b, b, b, 1.0);
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
            self.program,
            [
                # Map in_vert to the first 2 floats
                # Map in_color to the next 3 floats
                self.vbo.bind('in_vert', 'in_color', layout='2f 3f'),
            ],
        )

class Noise(BaseShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'scale' not in self.kwargs:
            self.kwargs['scale'] = 1.
        if 'speed' not in self.kwargs:
            self.kwargs['speed'] = 1.

    def get_fragment_shader(self):
        return """
            #version 330

            uniform float time;
            uniform vec2 resolution;

            uniform float scale;
            uniform float speed;

            out vec4 out_color;

            /* discontinuous pseudorandom uniformly distributed in [-0.5, +0.5]^3 */
            vec3 random3(vec3 c) {
                float j = 4096.0*sin(dot(c,vec3(17.0, 59.4, 15.0)));
                vec3 r;
                r.z = fract(512.0*j);
                j *= .125;
                r.x = fract(512.0*j);
                j *= .125;
                r.y = fract(512.0*j);
                return r-0.5;
            }

            /* skew constants for 3d simplex functions */
            const float F3 =  0.3333333;
            const float G3 =  0.1666667;

            /* 3d simplex noise */
            float simplex3d(vec3 p) {
                /* 1. find current tetrahedron T and it's four vertices */
                /* s, s+i1, s+i2, s+1.0 - absolute skewed (integer) coordinates of T vertices */
                /* x, x1, x2, x3 - unskewed coordinates of p relative to each of T vertices*/
                
                /* calculate s and x */
                vec3 s = floor(p + dot(p, vec3(F3)));
                vec3 x = p - s + dot(s, vec3(G3));
                
                /* calculate i1 and i2 */
                vec3 e = step(vec3(0.0), x - x.yzx);
                vec3 i1 = e*(1.0 - e.zxy);
                vec3 i2 = 1.0 - e.zxy*(1.0 - e);
                    
                /* x1, x2, x3 */
                vec3 x1 = x - i1 + G3;
                vec3 x2 = x - i2 + 2.0*G3;
                vec3 x3 = x - 1.0 + 3.0*G3;
                
                /* 2. find four surflets and store them in d */
                vec4 w, d;
                
                /* calculate surflet weights */
                w.x = dot(x, x);
                w.y = dot(x1, x1);
                w.z = dot(x2, x2);
                w.w = dot(x3, x3);
                
                /* w fades from 0.6 at the center of the surflet to 0.0 at the margin */
                w = max(0.6 - w, 0.0);
                
                /* calculate surflet components */
                d.x = dot(random3(s), x);
                d.y = dot(random3(s + i1), x1);
                d.z = dot(random3(s + i2), x2);
                d.w = dot(random3(s + 1.0), x3);
                
                /* multiply d by w^4 */
                w *= w;
                w *= w;
                d *= w;
                
                /* 3. return the sum of the four surflets */
                return dot(d, vec4(52.0));
            }

            /* const matrices for 3d rotation */
            const mat3 rot1 = mat3(-0.37, 0.36, 0.85,-0.14,-0.93, 0.34,0.92, 0.01,0.4);
            const mat3 rot2 = mat3(-0.55,-0.39, 0.74, 0.33,-0.91,-0.24,0.77, 0.12,0.63);
            const mat3 rot3 = mat3(-0.71, 0.52,-0.47,-0.08,-0.72,-0.68,-0.7,-0.45,0.56);

            /* directional artifacts can be reduced by rotating each octave */
            float simplex3d_fractal(vec3 m) {
                return   0.5333333*simplex3d(m*rot1)
                        +0.2666667*simplex3d(2.0*m*rot2)
                        +0.1333333*simplex3d(4.0*m*rot3)
                        +0.0666667*simplex3d(8.0*m);
            }

            void main(){

                ivec2 coord = ivec2(gl_FragCoord.xy);

                vec2 p = vec2(1.0) * coord.xy/resolution.xy;
                vec3 p3 = vec3(p * scale, time*0.025 * speed);
                
                float noise = simplex3d_fractal(p3*8.0+8.0);

                // scale to 0-1 for image storage
                noise = 0.5 + (0.5 * noise);

                out_color = vec4(vec3(noise), 1.);
            }
        """

class Add(BaseShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'amount' not in self.kwargs:
            self.kwargs['amount'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;
            uniform sampler2D other_texture;

            uniform float amount;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = texture(input_texture, uv).rgb;
                vec3 other = texture(other_texture, uv).rgb;
                color = mix(color, color + other, amount);
                fragColor = vec4(color, 1.0);
            }
            """
    
class Multiply(BaseShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'amount' not in self.kwargs:
            self.kwargs['amount'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;
            uniform sampler2D other_texture;

            uniform float amount;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = texture(input_texture, uv).rgb;
                vec3 other = texture(other_texture, uv).rgb;
                color = mix(color, color * other, amount);
                fragColor = vec4(color, 1.0);
            }
            """
    
class Modulate(BaseShader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'amount' not in self.kwargs:
            self.kwargs['amount'] = 0.5

    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;
            uniform sampler2D other_texture;

            uniform float amount;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 other = texture(other_texture, uv).rgb;
                uv += other.xy * amount;
                vec3 color = texture(input_texture, uv).rgb;
                fragColor = vec4(color, 1.0);
            }
            """

class Identity(BaseShader):
    def get_fragment_shader(self):
        return """
            #version 330

            uniform sampler2D input_texture;

            uniform vec2 resolution;
            uniform float time;

            out vec4 fragColor;

            void main() {
                vec2 uv = gl_FragCoord.xy / resolution.xy;
                vec3 color = texture(input_texture, uv).rgb;
                fragColor = vec4(color, 1.0);
            }
            """
    