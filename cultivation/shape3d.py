import json
import logging
import math

import glm
import numpy as np
import moderngl
import trimesh

from layers import BaseShader, BaseLayer
import utils


class Primitives(BaseLayer):
    """Layer for rendering multiple primitive shapes"""
    def __init__(self, ctx, logger: logging.Logger, **kwargs):
        self.primitives = []
        self.logger = logger
        self.ctx = ctx
        primitives_kwargs = kwargs.copy()
        primitives_kwargs.pop('shapes', None)

        for primitive in kwargs.get('shapes', []):
            primitive_kwargs = primitive | primitives_kwargs
            shape = Shape3D(ctx, logger, **primitive_kwargs)
            self.primitives.append(shape)
        self.source_layer = True

    def render(self, **kwargs):
        for primitive in self.primitives:
            try:
                primitive.render(**kwargs)
            except Exception as e:
                self.logger.error(f"Error rendering primitive {primitive}: {e}")
                # render black if error occurs
                self.ctx.clear(0.0, 0.0, 0.0, 1.0)

    def get_uniforms(self):
        uniforms = []
        for primitive in self.primitives:
            uniforms += primitive.get_uniforms()
        return uniforms

class Shape3D(BaseShader):
    """Layer for rendering 3D primitive shapes"""
    
    def __init__(self, ctx, logger: logging.Logger, **kwargs):
        self.shape_type = kwargs.get('shape_type', 'cube')
        self.size = kwargs.get('size', 1.0)
        self.position = kwargs.get('position', [0.0, 0.0, 0.0])
        self.rotation = kwargs.get('rotation', [0.0, 0.0, 0.0])
        self.color = kwargs.get('color', [1.0, 1.0, 1.0])
        
        # New lighting parameters
        self.use_lighting = kwargs.get('lighting', True)
        self.ambient_strength = kwargs.get('ambient_light', 0.2)
        self.specular_strength = kwargs.get('specular_light', 0.5)
        self.light_color = kwargs.get('light_color', [1.0, 1.0, 1.0])
        self.light_position = kwargs.get('light_position', [2.0, 2.0, 2.0])
        self.camera_position = kwargs.get('camera_position', [0.0, 0.0, 3.0])
        
        self.shape_detail = kwargs.get('detail', 16)  # For spheres, cylinders
        self.source_layer = True
        
        # Process parameters
        if not ((isinstance(self.size, list) and len(self.size) == 3) or isinstance(self.size, (int, float, str))):
            logger.warning(f"Invalid size format: {self.size}. Defaulting to 1.0")
            self.size = 1.0

        # Ensure position is a list of 3 elements
        if not isinstance(self.position, list) or len(self.position) != 3:
            logger.warning(f"Invalid position format: {self.position}. Defaulting to [0.0, 0.0, 0.0]")
            self.position = [0.0, 0.0, 0.0]
        
        # Ensure rotation is a list of 3 elements
        if not isinstance(self.rotation, list) or len(self.rotation) != 3:
            logger.warning(f"Invalid rotation format: {self.rotation}. Defaulting to [0.0, 0.0, 0.0]")
            self.rotation = [0.0, 0.0, 0.0]
        
        # Ensure color is a list of 3 elements
        if not (isinstance(self.color, str) or (isinstance(self.color, list) and len(self.color) == 3)):
            logger.warning(f"Invalid color format: {self.color}. Defaulting to [1.0, 1.0, 1.0]")
            self.color = [1.0, 1.0, 1.0]
        
        # Ensure light_color is a list of 3 elements if provided
        if not isinstance(self.light_color, list) or len(self.light_color) != 3:
            logger.warning(f"Invalid light_color format: {self.light_color}. Defaulting to [1.0, 1.0, 1.0]")
            self.light_color = [1.0, 1.0, 1.0]
        
        # Ensure light_position is a list of 3 elements if provided
        if not isinstance(self.light_position, list) or len(self.light_position) != 3:
            logger.warning(f"Invalid light_position format: {self.light_position}. Defaulting to [2.0, 2.0, 2.0]")
            self.light_position = [2.0, 2.0, 2.0]
            
        # Remove our custom params from kwargs before passing to parent
        for key in ['shape_type', 'size', 'position', 'rotation', 'color', 'lighting', 'detail',
                    'ambient_strength', 'specular_strength', 'light_color', 'light_position', 
                    'light_follow_camera', 'shininess', 'texture']:
            if key in kwargs:
                kwargs.pop(key)

        super().__init__(ctx, logger, **kwargs)
        
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
        
        real_camera = []
        for p in self.camera_position:
            if isinstance(p, str):
                # Handle camera position expressions
                p = utils.eval_statement(p, kwargs, 0.0, 'camera_position', self.logger)
            elif isinstance(p, int):
                p = float(p)
            real_camera.append(float(p))

        # Create view matrix (camera)
        view = glm.lookAt(
            glm.vec3(*real_camera),  # Camera position
            glm.vec3(0.0, 0.0, 0.0),  # Look at
            glm.vec3(0.0, 1.0, 0.0)   # Up vector
        )
        
        # Create model matrix (position/rotation of shape)
        model = glm.mat4(1.0)
        
        # Apply rotation (convert degrees to radians)
        real_rot = []
        for r in self.rotation:
            if isinstance(r, str):
                # Handle rotation expressions
                r = utils.eval_statement(r, kwargs, 0.0, 'rotation', self.logger)
            else:
                r = math.radians(float(r))
            real_rot.append(float(r))
        
        model = model * glm.rotate(glm.mat4(1.0), real_rot[0], glm.vec3(1.0, 0.0, 0.0))
        model = model * glm.rotate(glm.mat4(1.0), real_rot[1], glm.vec3(0.0, 1.0, 0.0))
        model = model * glm.rotate(glm.mat4(1.0), real_rot[2], glm.vec3(0.0, 0.0, 1.0))
        
        # Apply scale
        real_size = []
        if isinstance(self.size, int):
            real_size = map(float, [self.size, self.size, self.size])
        elif isinstance(self.size, str):
            # Handle size expressions
            real_size = utils.eval_statement(self.size, kwargs, 1.0, 'size', self.logger)
            real_size = [real_size, real_size, real_size]
        elif isinstance(self.size, list) and len(self.size) == 3:
            for s in self.size:
                if isinstance(s, str):
                    s = utils.eval_statement(s, kwargs, 1.0, 'size', self.logger)
                elif isinstance(s, int):
                    s = float(s)
                real_size.append(float(s))
        model = model * glm.scale(glm.mat4(1.0), glm.vec3(*real_size))
        
        # Apply position
        real_pos = []
        for p in self.position:
            if isinstance(p, str):
                # Handle position expressions
                p = utils.eval_statement(p, kwargs, 0.0, 'position', self.logger)
            elif isinstance(p, int):
                p = float(p)
            real_pos.append(float(p))
        
        model = model * glm.translate(glm.mat4(1.0), glm.vec3(*real_pos))

        # evaluate colour
        use_color_texture = False
        if isinstance(self.color, str):
            if self.color in kwargs:
                use_color_texture = True
            else:
                self.logger.warning(f"Color {self.color} not found in kwargs. Defaulting to [1.0, 1.0, 1.0]")
                real_color = [1.0, 1.0, 1.0]
        elif isinstance(self.color, list) and len(self.color) == 3:
            real_color = []
            for c in self.color:
                if isinstance(c, str):
                    # Handle color expressions
                    c = utils.eval_statement(c, kwargs, 1.0, 'color', self.logger)
                elif isinstance(c, int):
                    c = float(c)
                real_color.append(float(c))
        else:
            self.logger.warning(f"Invalid color format: {self.color}. Defaulting to [1.0, 1.0, 1.0]")
            real_color = [1.0, 1.0, 1.0]
        
        # Set uniforms
        if 'model_matrix' in self.program:
            self.program['model_matrix'].write(model)
        if 'view_matrix' in self.program:
            self.program['view_matrix'].write(view)
        if 'proj_matrix' in self.program:
            self.program['proj_matrix'].write(proj)
        if use_color_texture:
            self.program['use_texture'] = True
            self.program['texture_source'] = kwargs[self.color]
        else:
            if 'color' in self.program:
                self.program['color'] = tuple(real_color)
        if 'time' in self.program:
            self.program['time'] = time
        if 'use_lighting' in self.program:
            self.program['use_lighting'] = self.use_lighting
        
        uniform_names = ['ambient_strength', 'specular_strength', 'light_color', 'light_position']
        for name in uniform_names:
            if name in self.program:
                if name in ['light_position', 'light_color']:
                    # Handle light position as a list
                    self.program[name] = tuple(getattr(self, name))
                else:
                    # Handle other lighting parameters
                    self.program[name] = getattr(self, name)
        
        # Set other uniforms from kwargs
        for k, val in kwargs.items():
            if k in self.program and k not in ['resolution', 'time', 'fft']:
                self.program[k] = val
        
        # Set any custom parameters from self.kwargs
        for k, val in self.kwargs.items():
            if k in self.program:
                if isinstance(val, str):
                    self.program[k] = utils.eval_statement(val, kwargs, 1.0, k, self.logger)
                else:
                    self.program[k] = val
        
        # Render the shape
        self.vao.render()
        
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
        out vec3 local_pos;

        void main() {
            v_normal = mat3(model_matrix) * in_normal;
            vec4 world_pos = model_matrix * vec4(in_position, 1.0);
            v_position = world_pos.xyz;
            local_pos = in_position; // Assigning in_position to local_pos
            gl_Position = proj_matrix * view_matrix * world_pos;
        }
        """
    
    def get_fragment_shader(self):
        return """
            #version 330

            uniform vec3 color;
            uniform bool use_lighting;
            uniform float time;

            // Lighting uniforms
            uniform float ambient_strength;
            uniform float specular_strength;
            uniform vec3 light_color;
            uniform vec3 light_position;
            uniform float shininess;

            // Texture uniforms
            uniform bool use_texture;
            uniform sampler2D texture_source;

            in vec3 v_normal;
            in vec3 v_position;
            in vec3 local_pos;

            out vec4 frag_color;

            void main() {
                vec3 base_color = color; // Initialize base_color with color

                // Apply texture if available
                if (use_texture) {
                    vec2 uv = local_pos.xy + 0.5;  // Simple UV mapping
                    base_color = texture(texture_source, uv).rgb;
                } else {
                    base_color = color; // Ensure base_color is set to color if not using texture
                }
                
                if (use_lighting) {
                    // Use the provided light position
                    vec3 light_pos = light_position;
                    
                    // Ambient
                    vec3 ambient = ambient_strength * light_color;
                    
                    // Diffuse
                    vec3 norm = normalize(v_normal);
                    vec3 light_dir = normalize(light_pos - v_position);
                    float diff = max(dot(norm, light_dir), 0.0);
                    vec3 diffuse = diff * light_color;
                    
                    // Specular
                    vec3 view_dir = normalize(vec3(0.0, 0.0, 3.0) - v_position);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
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