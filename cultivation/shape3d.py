import numpy as np
import moderngl
import glm
import json
import math

from layers import BaseShader

class Shape3D(BaseShader):
    """Layer for rendering 3D primitive shapes"""
    
    def __init__(self, ctx, **kwargs):
        self.shape_type = kwargs.get('shape_type', 'cube')
        self.size = kwargs.get('size', 1.0)
        self.position = kwargs.get('position', [0.0, 0.0, 0.0])
        self.rotation = kwargs.get('rotation', [0.0, 0.0, 0.0])
        self.color = kwargs.get('color', [1.0, 1.0, 1.0])
        self.use_lighting = kwargs.get('lighting', True)
        self.shape_detail = kwargs.get('detail', 16)  # For spheres, cylinders
        self.source_layer = True
        
        # Process parameters
        # Ensure position is a list of 3 elements
        if not isinstance(self.position, list) or len(self.position) != 3:
            self.position = [0.0, 0.0, 0.0]
        
        # Ensure rotation is a list of 3 elements
        if not isinstance(self.rotation, list) or len(self.rotation) != 3:
            self.rotation = [0.0, 0.0, 0.0]
        
        # Ensure color is a list of 3 elements
        if not isinstance(self.color, list) or len(self.color) != 3:
            self.color = [1.0, 1.0, 1.0]
            
        # Remove our custom params from kwargs before passing to parent
        for key in ['shape_type', 'size', 'position', 'rotation', 'color', 'lighting', 'detail']:
            if key in kwargs:
                kwargs.pop(key)
            
        super().__init__(ctx, **kwargs)
        
    def load_vao(self):
        # Generate shape vertices based on type
        if self.shape_type == 'cube':
            vertices, indices = self._generate_cube()
        elif self.shape_type == 'sphere':
            vertices, indices = self._generate_sphere(self.shape_detail)
        elif self.shape_type == 'cylinder':
            vertices, indices = self._generate_cylinder(self.shape_detail)
        elif self.shape_type == 'plane':
            vertices, indices = self._generate_plane()
        elif self.shape_type == 'torus':
            vertices, indices = self._generate_torus(self.shape_detail)
        else:
            # Default to cube if unknown
            vertices, indices = self._generate_cube()
            
        # Load vertices and indices into buffers
        self.vbo = self.ctx.buffer(vertices.astype('f4'))
        self.ibo = self.ctx.buffer(indices.astype('i4'))
        
        # Create vertex array with position, normal attributes
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f 3f', 'in_position', 'in_normal')
            ],
            self.ibo
        )
        
    def render(self, **kwargs):
        # Set uniforms from kwargs
        resolution = kwargs.get('resolution', (1280, 720))
        time = kwargs.get('time', 0.0)
        
        aspect_ratio = resolution[0] / resolution[1]
        
        # Create projection matrix (perspective)
        proj = glm.perspective(math.radians(45.0), aspect_ratio, 0.1, 100.0)
        
        # Create view matrix (camera)
        view = glm.lookAt(
            glm.vec3(0.0, 0.0, 3.0),  # Camera position
            glm.vec3(0.0, 0.0, 0.0),  # Look at
            glm.vec3(0.0, 1.0, 0.0)   # Up vector
        )
        
        # Create model matrix (position/rotation of shape)
        model = glm.mat4(1.0)
        
        # Apply rotation (convert degrees to radians)
        rot_x, rot_y, rot_z = [math.radians(float(r)) if not isinstance(r, str) else r for r in self.rotation]
        
        # Handle rotation expressions
        if isinstance(rot_x, str):
            try:
                rot_x = eval(rot_x, {}, {'time': time, 'np': np, 'math': math, 'sin': math.sin, 'cos': math.cos})
            except Exception as e:
                print(f"Error evaluating rotation expression: {e}")
                rot_x = 0.0
                
        if isinstance(rot_y, str):
            try:
                rot_y = eval(rot_y, {}, {'time': time, 'np': np, 'math': math, 'sin': math.sin, 'cos': math.cos})
            except Exception as e:
                print(f"Error evaluating rotation expression: {e}")
                rot_y = 0.0
                
        if isinstance(rot_z, str):
            try:
                rot_z = eval(rot_z, {}, {'time': time, 'np': np, 'math': math, 'sin': math.sin, 'cos': math.cos})
            except Exception as e:
                print(f"Error evaluating rotation expression: {e}")
                rot_z = 0.0
        
        model = model * glm.rotate(glm.mat4(1.0), rot_x, glm.vec3(1.0, 0.0, 0.0))
        model = model * glm.rotate(glm.mat4(1.0), rot_y, glm.vec3(0.0, 1.0, 0.0))
        model = model * glm.rotate(glm.mat4(1.0), rot_z, glm.vec3(0.0, 0.0, 1.0))
        
        # Apply scale
        size = self.size if isinstance(self.size, (int, float)) else 1.0
        model = model * glm.scale(glm.mat4(1.0), glm.vec3(size, size, size))
        
        # Apply position
        x, y, z = self.position
        
        # Handle position expressions
        if isinstance(x, str):
            try:
                x = eval(x, {}, {'time': time, 'np': np, 'math': math, 'sin': math.sin, 'cos': math.cos})
            except Exception as e:
                print(f"Error evaluating position expression: {e}")
                x = 0.0
                
        if isinstance(y, str):
            try:
                y = eval(y, {}, {'time': time, 'np': np, 'math': math, 'sin': math.sin, 'cos': math.cos})
            except Exception as e:
                print(f"Error evaluating position expression: {e}")
                y = 0.0
                
        if isinstance(z, str):
            try:
                z = eval(z, {}, {'time': time, 'np': np, 'math': math, 'sin': math.sin, 'cos': math.cos})
            except Exception as e:
                print(f"Error evaluating position expression: {e}")
                z = 0.0
        
        model = model * glm.translate(glm.mat4(1.0), glm.vec3(x, y, z))
        
        # Set uniforms
        if 'model_matrix' in self.program:
            self.program['model_matrix'].write(model)
        if 'view_matrix' in self.program:
            self.program['view_matrix'].write(view)
        if 'proj_matrix' in self.program:
            self.program['proj_matrix'].write(proj)
        if 'color' in self.program:
            self.program['color'] = tuple(self.color)
        if 'time' in self.program:
            self.program['time'] = time
        if 'use_lighting' in self.program:
            self.program['use_lighting'] = self.use_lighting
        
        # Set other uniforms from kwargs
        for k, val in kwargs.items():
            if k in self.program and k not in ['resolution', 'time', 'fft']:
                self.program[k] = val
        
        # Set any custom parameters from self.kwargs
        for k, val in self.kwargs.items():
            if k in self.program:
                if isinstance(val, str):
                    try:
                        self.program[k] = eval(val, {}, {'fft': kwargs.get('fft', None), 'np': np, 'time': kwargs.get('time', 0.0)})
                    except Exception as e:
                        print(f"Error setting uniform {k}: {e}")
                else:
                    self.program[k] = val
        
        # Render the shape
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.vao.render()
        self.ctx.disable(moderngl.DEPTH_TEST)
        
    def get_vertex_shader(self):
        return """
        #version 330

        uniform mat4 model_matrix;
        uniform mat4 view_matrix;
        uniform mat4 proj_matrix;
        uniform float time;

        in vec3 in_position;
        in vec3 in_normal;

        out vec3 v_normal;
        out vec3 v_position;

        void main() {
            v_normal = mat3(model_matrix) * in_normal;
            vec4 world_pos = model_matrix * vec4(in_position, 1.0);
            v_position = world_pos.xyz;
            gl_Position = proj_matrix * view_matrix * world_pos;
        }
        """
    
    def get_fragment_shader(self):
        return """
        #version 330

        uniform vec3 color;
        uniform bool use_lighting;
        uniform float time;

        in vec3 v_normal;
        in vec3 v_position;
        
        out vec4 frag_color;

        void main() {
            vec3 base_color = color;
            
            if (use_lighting) {
                // Simple lighting calculation
                vec3 light_pos = vec3(2.0 * sin(time), 2.0, 2.0 * cos(time));
                vec3 light_color = vec3(1.0, 1.0, 1.0);
                
                // Ambient
                float ambient_strength = 0.2;
                vec3 ambient = ambient_strength * light_color;
                
                // Diffuse
                vec3 norm = normalize(v_normal);
                vec3 light_dir = normalize(light_pos - v_position);
                float diff = max(dot(norm, light_dir), 0.0);
                vec3 diffuse = diff * light_color;
                
                // Specular
                float specular_strength = 0.5;
                vec3 view_dir = normalize(vec3(0.0, 0.0, 3.0) - v_position);
                vec3 reflect_dir = reflect(-light_dir, norm);
                float spec = pow(max(dot(view_dir, reflect_dir), 0.0), 32);
                vec3 specular = specular_strength * spec * light_color;
                
                base_color = (ambient + diffuse + specular) * base_color;
            }
            
            frag_color = vec4(base_color, 1.0);
        }
        """
    
    def _generate_cube(self):
        # Vertices (position + normal) for a cube centered at origin
        vertices = np.array([
            # Front face
            -0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  # Bottom-left
             0.5, -0.5,  0.5,  0.0,  0.0,  1.0,  # Bottom-right
             0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  # Top-right
            -0.5,  0.5,  0.5,  0.0,  0.0,  1.0,  # Top-left
            
            # Back face
            -0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  # Bottom-left
             0.5, -0.5, -0.5,  0.0,  0.0, -1.0,  # Bottom-right
             0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  # Top-right
            -0.5,  0.5, -0.5,  0.0,  0.0, -1.0,  # Top-left
            
            # Right face
             0.5, -0.5, -0.5,  1.0,  0.0,  0.0,  # Bottom-left
             0.5, -0.5,  0.5,  1.0,  0.0,  0.0,  # Bottom-right
             0.5,  0.5,  0.5,  1.0,  0.0,  0.0,  # Top-right
             0.5,  0.5, -0.5,  1.0,  0.0,  0.0,  # Top-left
            
            # Left face
            -0.5, -0.5, -0.5, -1.0,  0.0,  0.0,  # Bottom-left
            -0.5, -0.5,  0.5, -1.0,  0.0,  0.0,  # Bottom-right
            -0.5,  0.5,  0.5, -1.0,  0.0,  0.0,  # Top-right
            -0.5,  0.5, -0.5, -1.0,  0.0,  0.0,  # Top-left
            
            # Top face
            -0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  # Bottom-left
             0.5,  0.5,  0.5,  0.0,  1.0,  0.0,  # Bottom-right
             0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  # Top-right
            -0.5,  0.5, -0.5,  0.0,  1.0,  0.0,  # Top-left
            
            # Bottom face
            -0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  # Bottom-left
             0.5, -0.5,  0.5,  0.0, -1.0,  0.0,  # Bottom-right
             0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  # Top-right
            -0.5, -0.5, -0.5,  0.0, -1.0,  0.0,  # Top-left
        ], dtype='f4')
        
        # Indices for drawing the cube using triangles
        indices = np.array([
            0, 1, 2, 2, 3, 0,     # Front face
            4, 5, 6, 6, 7, 4,     # Back face
            8, 9, 10, 10, 11, 8,  # Right face
            12, 13, 14, 14, 15, 12,  # Left face
            16, 17, 18, 18, 19, 16,  # Top face
            20, 21, 22, 22, 23, 20,  # Bottom face
        ], dtype='i4')
        
        return vertices, indices
    
    def _generate_sphere(self, segments=16):
        """Generate a UV sphere with the given number of segments"""
        vertices = []
        indices = []
        
        # Generate vertices
        for i in range(segments + 1):
            for j in range(segments):
                theta = i * np.pi / segments
                phi = j * 2 * np.pi / segments
                
                x = np.sin(theta) * np.cos(phi)
                y = np.cos(theta)
                z = np.sin(theta) * np.sin(phi)
                
                # Position and normal (same for sphere)
                vertices.extend([x, y, z, x, y, z])
        
        # Generate indices
        for i in range(segments):
            for j in range(segments):
                first = i * (segments) + j
                second = first + segments
                
                indices.extend([first, second, first + 1])
                indices.extend([second, second + 1, first + 1])
        
        return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')
    
    def _generate_cylinder(self, segments=16):
        """Generate a cylinder with the given number of segments"""
        vertices = []
        indices = []
        
        # Generate circle points
        for i in range(segments):
            angle = i * 2 * np.pi / segments
            x = np.cos(angle)
            z = np.sin(angle)
            
            # Top circle vertex
            nx = x
            ny = 0.0
            nz = z
            vertices.extend([x, 0.5, z, nx, ny, nz])
            
            # Bottom circle vertex
            vertices.extend([x, -0.5, z, nx, ny, nz])
            
            # Top cap normal
            vertices.extend([x, 0.5, z, 0.0, 1.0, 0.0])
            
            # Bottom cap normal
            vertices.extend([x, -0.5, z, 0.0, -1.0, 0.0])
        
        # Center points for caps
        vertices.extend([0.0, 0.5, 0.0, 0.0, 1.0, 0.0])  # Top center
        vertices.extend([0.0, -0.5, 0.0, 0.0, -1.0, 0.0])  # Bottom center
        
        top_center_idx = len(vertices) // 6 - 2
        bottom_center_idx = len(vertices) // 6 - 1
        
        # Generate indices for the sides
        for i in range(segments):
            i2 = i * 4  # Each point has 4 vertices (side top, side bottom, cap top, cap bottom)
            i2_next = ((i + 1) % segments) * 4
            
            # Side triangles
            indices.extend([i2, i2 + 1, i2_next])
            indices.extend([i2_next, i2 + 1, i2_next + 1])
            
            # Top cap triangles
            indices.extend([i2 + 2, top_center_idx, i2_next + 2])
            
            # Bottom cap triangles
            indices.extend([i2 + 3, i2_next + 3, bottom_center_idx])
        
        return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')
    
    def _generate_plane(self):
        """Generate a simple plane on the XZ plane"""
        vertices = np.array([
            # Position (x, y, z), Normal (nx, ny, nz)
            -0.5, 0.0, -0.5,  0.0, 1.0, 0.0,
             0.5, 0.0, -0.5,  0.0, 1.0, 0.0,
             0.5, 0.0,  0.5,  0.0, 1.0, 0.0,
            -0.5, 0.0,  0.5,  0.0, 1.0, 0.0,
        ], dtype='f4')
        
        indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype='i4')
        
        return vertices, indices
    
    def _generate_torus(self, segments=16, tube_segments=8, radius=0.3, tube_radius=0.1):
        """Generate a torus (donut shape)"""
        vertices = []
        indices = []
        
        for i in range(segments):
            for j in range(tube_segments):
                u = i * 2 * np.pi / segments
                v = j * 2 * np.pi / tube_segments
                
                # Calculate position
                x = (radius + tube_radius * np.cos(v)) * np.cos(u)
                y = tube_radius * np.sin(v)
                z = (radius + tube_radius * np.cos(v)) * np.sin(u)
                
                # Calculate normal
                cx = radius * np.cos(u)
                cz = radius * np.sin(u)
                
                nx = x - cx
                ny = y
                nz = z - cz
                
                # Normalize normal
                norm = np.sqrt(nx*nx + ny*ny + nz*nz)
                nx /= norm
                ny /= norm
                nz /= norm
                
                vertices.extend([x, y, z, nx, ny, nz])
        
        # Generate indices
        for i in range(segments):
            for j in range(tube_segments):
                # Calculate indices
                p0 = i * tube_segments + j
                p1 = ((i + 1) % segments) * tube_segments + j
                p2 = ((i + 1) % segments) * tube_segments + (j + 1) % tube_segments
                p3 = i * tube_segments + (j + 1) % tube_segments
                
                # Add triangles
                indices.extend([p0, p1, p2])
                indices.extend([p0, p2, p3])
        
        return np.array(vertices, dtype='f4'), np.array(indices, dtype='i4')
    
    def __str__(self) -> str:
        return f"Shape3D({self.shape_type}, size={self.size}, pos={self.position})"
    
class Cube(Shape3D):
    def __init__(self, *args, **kwargs):
        if 'shape_type' not in kwargs:
            kwargs['shape_type'] = 'cube'
        super().__init__(*args, **kwargs)

class Cylinder(Shape3D):
    def __init__(self, *args, **kwargs):
        if 'shape_type' not in kwargs:
            kwargs['shape_type'] = 'cylinder'
        super().__init__(*args, **kwargs)

class Sphere(Shape3D):
    def __init__(self, *args, **kwargs):
        if 'shape_type' not in kwargs:
            kwargs['shape_type'] = 'sphere'
        super().__init__(*args, **kwargs)

class Plane(Shape3D):
    def __init__(self, *args, **kwargs):
        if 'shape_type' not in kwargs:
            kwargs['shape_type'] = 'plane'
        super().__init__(*args, **kwargs)

class Torus(Shape3D):
    def __init__(self, *args, **kwargs):
        if 'shape_type' not in kwargs:
            kwargs['shape_type'] = 'torus'
        super().__init__(*args, **kwargs)