import json
import logging
import math

import glm
import moderngl_window.geometry
import numpy as np
import moderngl
import moderngl_window
import trimesh

from layers import BaseShader, BaseLayer, Identity
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
            primitive_kwargs['shape_type'] = primitive.get('name', 'cube')
            shape = Shape3D(ctx, logger, **primitive_kwargs)
            self.primitives.append(shape)
        self.source_layer = True

    def render(self, **kwargs):
        final_layer = Identity(self.ctx, self.logger)
        final_layer.render(**kwargs)

        for primitive in self.primitives:
            try:
                primitive.render(**kwargs)
            except Exception as e:
                self.logger.error(f"Error rendering primitive {primitive}: {e}")

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
        self.shininess = kwargs.get('shininess', 32.0)
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
                    'light_follow_camera', 'shininess', 'texture', 'camera_position']:
            if key in kwargs:
                kwargs.pop(key)

        super().__init__(ctx, logger, **kwargs)
        
    def load_vao(self):
        # Get vertices based on shape type
        if self.shape_type == 'cube':
            vao = moderngl_window.geometry.cube()
        elif self.shape_type == 'sphere':
            vao = moderngl_window.geometry.sphere()
        elif self.shape_type == 'cylinder':
            raise NotImplementedError("Cylinder shape generation is not implemented yet.")
            vertices, indices = self._generate_cylinder(self.shape_detail)
        elif self.shape_type == 'plane':
            vao = moderngl_window.geometry.quad_2d()
        elif self.shape_type == 'torus':
            raise NotImplementedError("Torus shape generation is not implemented yet.")
            vertices, indices = self._generate_torus(self.shape_detail)
        else:
            # Default to cube if unknown
            vao = moderngl_window.geometry.cube()
            
        # Load vertices into buffer
        self.vao = vao
        
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
        if 'camera_position' in self.program:
            self.program['camera_position'].write(glm.vec3(*real_camera))
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
        
        uniform_names = ['ambient_strength', 'specular_strength', 'shininess', 'light_color', 'light_position']
        for name in uniform_names:
            if name in self.program:
                if name in ['light_position', 'light_color']:
                    # Handle light position as a list
                    param = getattr(self, name)
                    real_param = []
                    for p in param:
                        if isinstance(p, str):
                            # Handle light position expressions
                            p = utils.eval_statement(p, kwargs, 0.0, name, self.logger)
                        elif isinstance(p, int):
                            p = float(p)
                        real_param.append(float(p))
                    self.program[name] = tuple(real_param)
                else:
                    # Handle other lighting parameters
                    param = getattr(self, name)
                    if isinstance(param, str):
                        # Handle lighting parameter expressions
                        param = utils.eval_statement(param, kwargs, 1.0, name, self.logger)
                    self.program[name] = param
        
        # Set other uniforms from kwargs
        for k, val in kwargs.items():
            if k in self.program and k not in ['resolution', 'time', 'fft']:
                self.program[k] = val
        
        # Set any custom parameters from self.kwargs
        for k, val in self.kwargs.items():
            if k in self.program:
                if isinstance(val, str):
                    self.program[k] = utils.eval_statement(val, kwargs, 1.0, k, self.logger)
                elif isinstance(val, list):
                    real_val = []
                    for v in val:
                        if isinstance(v, str):
                            # Handle custom parameter expressions
                            v = utils.eval_statement(v, kwargs, 1.0, k, self.logger)
                        elif isinstance(v, int):
                            v = float(v)
                        real_val.append(float(v))
                    self.program[k] = tuple(real_val)
                else:
                    self.program[k] = val

        # Render the shape
        self.vao.render(self.program)
        
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
            vec4 world_normal = model_matrix * vec4(in_normal, 1.0);
            v_normal = world_normal.xyz;
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
            uniform vec3 camera_position;
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
                    vec3 view_dir = normalize(camera_position - v_position);
                    vec3 reflect_dir = reflect(-light_dir, norm);
                    float spec = pow(max(dot(view_dir, reflect_dir), 0.0), shininess);
                    vec3 specular = specular_strength * spec * light_color;
                    
                    base_color = (ambient + diffuse + specular) * base_color;
                }
                
                frag_color = vec4(base_color, 1.0);
            }
        """
    
    def _generate_cube(self):
        """Generate a cube with 36 vertices (no indexing)"""
        # Position and normal data separate rather than interleaved

        center = [0.0, 0.0, 0.0]
        width, height, depth = 1, 1, 1

        pos = np.array([
            center[0] + width, center[1] - height, center[2] + depth,
            center[0] + width, center[1] + height, center[2] + depth,
            center[0] - width, center[1] - height, center[2] + depth,
            center[0] + width, center[1] + height, center[2] + depth,
            center[0] - width, center[1] + height, center[2] + depth,
            center[0] - width, center[1] - height, center[2] + depth,
            center[0] + width, center[1] - height, center[2] - depth,
            center[0] + width, center[1] + height, center[2] - depth,
            center[0] + width, center[1] - height, center[2] + depth,
            center[0] + width, center[1] + height, center[2] - depth,
            center[0] + width, center[1] + height, center[2] + depth,
            center[0] + width, center[1] - height, center[2] + depth,
            center[0] + width, center[1] - height, center[2] - depth,
            center[0] + width, center[1] - height, center[2] + depth,
            center[0] - width, center[1] - height, center[2] + depth,
            center[0] + width, center[1] - height, center[2] - depth,
            center[0] - width, center[1] - height, center[2] + depth,
            center[0] - width, center[1] - height, center[2] - depth,
            center[0] - width, center[1] - height, center[2] + depth,
            center[0] - width, center[1] + height, center[2] + depth,
            center[0] - width, center[1] + height, center[2] - depth,
            center[0] - width, center[1] - height, center[2] + depth,
            center[0] - width, center[1] + height, center[2] - depth,
            center[0] - width, center[1] - height, center[2] - depth,
            center[0] + width, center[1] + height, center[2] - depth,
            center[0] + width, center[1] - height, center[2] - depth,
            center[0] - width, center[1] - height, center[2] - depth,
            center[0] + width, center[1] + height, center[2] - depth,
            center[0] - width, center[1] - height, center[2] - depth,
            center[0] - width, center[1] + height, center[2] - depth,
            center[0] + width, center[1] + height, center[2] - depth,
            center[0] - width, center[1] + height, center[2] - depth,
            center[0] + width, center[1] + height, center[2] + depth,
            center[0] - width, center[1] + height, center[2] - depth,
            center[0] - width, center[1] + height, center[2] + depth,
            center[0] + width, center[1] + height, center[2] + depth,
        ], dtype=np.float32)

        normal_data = np.array([
            -0, 0, 1,
            -0, 0, 1,
            -0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            0, 0, 1,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0,
            1, 0, 0,
            0, -1, 0,
            0, -1, 0,
            0, -1, 0,
            0, -1, 0,
            0, -1, 0,
            0, -1, 0,
            -1, -0, 0,
            -1, -0, 0,
            -1, -0, 0,
            -1, -0, 0,
            -1, -0, 0,
            -1, -0, 0,
            0, 0, -1,
            0, 0, -1,
            0, 0, -1,
            0, 0, -1,
            0, 0, -1,
            0, 0, -1,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
            0, 1, 0,
        ], dtype=np.float32)
        
        # # Interleave position and normal data
        # vertices = np.zeros(36 * 6, dtype='f4')
        # for i in range(36):
        #     vertices[i*6:i*6+3] = positions[i*3:i*3+3]  # Position
        #     vertices[i*6+3:i*6+6] = normals[i*3:i*3+3]  # Normal
        
        # # No indices needed - we're using direct rendering
        indices = np.array([], dtype='i4')
        # Interleave position and normal data
        return np.stack([pos.reshape((-1, 3)), normal_data.reshape((-1, 3))], axis=-1), indices
    
    def _generate_sphere(self, segments=16):
        """Generate a UV sphere with direct rendering approach"""
        vertices = []
        
        # Generate vertices in triangles
        for i in range(segments):
            for j in range(segments):
                # Calculate the four corners of a quad on the sphere
                theta1 = i * np.pi / segments
                theta2 = (i + 1) * np.pi / segments
                phi1 = j * 2 * np.pi / segments
                phi2 = (j + 1) * 2 * np.pi / segments
                
                # Vertex 1 (top-left)
                x1 = np.sin(theta1) * np.cos(phi1)
                y1 = np.cos(theta1)
                z1 = np.sin(theta1) * np.sin(phi1)
                
                # Vertex 2 (bottom-left)
                x2 = np.sin(theta2) * np.cos(phi1)
                y2 = np.cos(theta2)
                z2 = np.sin(theta2) * np.sin(phi1)
                
                # Vertex 3 (bottom-right)
                x3 = np.sin(theta2) * np.cos(phi2)
                y3 = np.cos(theta2)
                z3 = np.sin(theta2) * np.sin(phi2)
                
                # Vertex 4 (top-right)
                x4 = np.sin(theta1) * np.cos(phi2)
                y4 = np.cos(theta1)
                z4 = np.sin(theta1) * np.sin(phi2)
                
                # Add first triangle (top-left, bottom-left, bottom-right)
                vertices.extend([x1, y1, z1, x1, y1, z1])  # position, normal
                vertices.extend([x2, y2, z2, x2, y2, z2])
                vertices.extend([x3, y3, z3, x3, y3, z3])
                
                # Add second triangle (top-left, bottom-right, top-right)
                vertices.extend([x1, y1, z1, x1, y1, z1])
                vertices.extend([x3, y3, z3, x3, y3, z3])
                vertices.extend([x4, y4, z4, x4, y4, z4])
        
        # For a sphere, the position and normal are the same (normalized)
        return np.array(vertices, dtype='f4'), np.array([], dtype='i4')

    def _generate_cylinder(self, segments=16):
        """Generate a cylinder with direct rendering approach"""
        vertices = []
        
        # 1. Generate the side vertices and triangles
        for i in range(segments):
            angle1 = i * 2 * np.pi / segments
            angle2 = ((i + 1) % segments) * 2 * np.pi / segments
            
            # Calculate positions
            x1 = np.cos(angle1)  # Top point 1
            z1 = np.sin(angle1)
            y1 = 0.5
            
            x2 = np.cos(angle1)  # Bottom point 1
            z2 = np.sin(angle1)
            y2 = -0.5
            
            x3 = np.cos(angle2)  # Bottom point 2
            z3 = np.sin(angle2)
            y3 = -0.5
            
            x4 = np.cos(angle2)  # Top point 2
            z4 = np.sin(angle2)
            y4 = 0.5
            
            # Calculate normals for sides (pointing outward)
            nx1, ny1, nz1 = x1, 0.0, z1  # Normal for point 1
            length1 = np.sqrt(nx1*nx1 + nz1*nz1)
            nx1, nz1 = nx1/length1, nz1/length1
            
            nx2, ny2, nz2 = x3, 0.0, z3  # Normal for point 2
            length2 = np.sqrt(nx2*nx2 + nz2*nz2)
            nx2, nz2 = nx2/length2, nz2/length2
            
            # First triangle of the side quad
            vertices.extend([x1, y1, z1, nx1, ny1, nz1])  # Top 1
            vertices.extend([x2, y2, z2, nx1, ny1, nz1])  # Bottom 1
            vertices.extend([x3, y3, z3, nx2, ny2, nz2])  # Bottom 2
            
            # Second triangle of the side quad
            vertices.extend([x1, y1, z1, nx1, ny1, nz1])  # Top 1
            vertices.extend([x3, y3, z3, nx2, ny2, nz2])  # Bottom 2
            vertices.extend([x4, y4, z4, nx2, ny2, nz2])  # Top 2
        
        # 2. Top cap (y = 0.5)
        for i in range(segments):
            angle1 = i * 2 * np.pi / segments
            angle2 = ((i + 1) % segments) * 2 * np.pi / segments
            
            x1 = 0.0           # Center
            y1 = 0.5
            z1 = 0.0
            
            x2 = np.cos(angle1)  # Edge point 1
            y2 = 0.5
            z2 = np.sin(angle1)
            
            x3 = np.cos(angle2)  # Edge point 2
            y3 = 0.5
            z3 = np.sin(angle2)
            
            # Top cap normals point up
            nx, ny, nz = 0.0, 1.0, 0.0
            
            # Add triangle
            vertices.extend([x1, y1, z1, nx, ny, nz])  # Center
            vertices.extend([x2, y2, z2, nx, ny, nz])  # Edge 1
            vertices.extend([x3, y3, z3, nx, ny, nz])  # Edge 2
        
        # 3. Bottom cap (y = -0.5)
        for i in range(segments):
            angle1 = i * 2 * np.pi / segments
            angle2 = ((i + 1) % segments) * 2 * np.pi / segments
            
            x1 = 0.0           # Center
            y1 = -0.5
            z1 = 0.0
            
            x2 = np.cos(angle1)  # Edge point 1
            y2 = -0.5
            z2 = np.sin(angle1)
            
            x3 = np.cos(angle2)  # Edge point 2
            y3 = -0.5
            z3 = np.sin(angle2)
            
            # Bottom cap normals point down
            nx, ny, nz = 0.0, -1.0, 0.0
            
            # Add triangle with reverse winding for bottom
            vertices.extend([x1, y1, z1, nx, ny, nz])  # Center
            vertices.extend([x3, y3, z3, nx, ny, nz])  # Edge 2
            vertices.extend([x2, y2, z2, nx, ny, nz])  # Edge 1
        
        return np.array(vertices, dtype='f4'), np.array([], dtype='i4')

    def _generate_plane(self):
        """Generate a plane on the XZ plane with direct rendering"""
        # Create a simple plane with 2 triangles
        vertices = [
            # Vertex positions            # Normals (pointing up)
            -0.5, 0.0, -0.5,              0.0, 1.0, 0.0,  # Back-left
            0.5, 0.0, -0.5,              0.0, 1.0, 0.0,  # Back-right
            0.5, 0.0,  0.5,              0.0, 1.0, 0.0,  # Front-right
            
            -0.5, 0.0, -0.5,              0.0, 1.0, 0.0,  # Back-left
            0.5, 0.0,  0.5,              0.0, 1.0, 0.0,  # Front-right
            -0.5, 0.0,  0.5,              0.0, 1.0, 0.0,  # Front-left
        ]
        
        return np.array(vertices, dtype='f4'), np.array([], dtype='i4')

    def _generate_torus(self, segments=16, tube_segments=8, radius=0.3, tube_radius=0.1):
        """Generate a torus with direct rendering"""
        vertices = []
        
        # Generate triangles for the torus surface
        for i in range(segments):
            for j in range(tube_segments):
                # Calculate indices for the 4 corners of a quad
                i1 = i
                i2 = (i + 1) % segments
                j1 = j
                j2 = (j + 1) % tube_segments
                
                # Calculate the 4 points of the quad
                # Point 1 (top-left)
                u1 = i1 * 2 * np.pi / segments
                v1 = j1 * 2 * np.pi / tube_segments
                
                # Point 2 (bottom-left)
                u2 = i2 * 2 * np.pi / segments
                v2 = j1 * 2 * np.pi / tube_segments
                
                # Point 3 (bottom-right)
                u3 = i2 * 2 * np.pi / segments
                v3 = j2 * 2 * np.pi / tube_segments
                
                # Point 4 (top-right)
                u4 = i1 * 2 * np.pi / segments
                v4 = j2 * 2 * np.pi / tube_segments
                
                # Calculate positions
                # Point 1
                cx1 = radius * np.cos(u1)
                cz1 = radius * np.sin(u1)
                x1 = cx1 + tube_radius * np.cos(v1) * np.cos(u1)
                y1 = tube_radius * np.sin(v1)
                z1 = cz1 + tube_radius * np.cos(v1) * np.sin(u1)
                
                # Point 2
                cx2 = radius * np.cos(u2)
                cz2 = radius * np.sin(u2)
                x2 = cx2 + tube_radius * np.cos(v2) * np.cos(u2)
                y2 = tube_radius * np.sin(v2)
                z2 = cz2 + tube_radius * np.cos(v2) * np.sin(u2)
                
                # Point 3
                cx3 = radius * np.cos(u3)
                cz3 = radius * np.sin(u3)
                x3 = cx3 + tube_radius * np.cos(v3) * np.cos(u3)
                y3 = tube_radius * np.sin(v3)
                z3 = cz3 + tube_radius * np.cos(v3) * np.sin(u3)
                
                # Point 4
                cx4 = radius * np.cos(u4)
                cz4 = radius * np.sin(u4)
                x4 = cx4 + tube_radius * np.cos(v4) * np.cos(u4)
                y4 = tube_radius * np.sin(v4)
                z4 = cz4 + tube_radius * np.cos(v4) * np.sin(u4)
                
                # Calculate normals (pointing outward from tube center)
                # Normal 1
                nx1 = x1 - cx1
                ny1 = y1
                nz1 = z1 - cz1
                length1 = np.sqrt(nx1*nx1 + ny1*ny1 + nz1*nz1)
                nx1, ny1, nz1 = nx1/length1, ny1/length1, nz1/length1
                
                # Normal 2
                nx2 = x2 - cx2
                ny2 = y2
                nz2 = z2 - cz2
                length2 = np.sqrt(nx2*nx2 + ny2*ny2 + nz2*nz2)
                nx2, ny2, nz2 = nx2/length2, ny2/length2, nz2/length2
                
                # Normal 3
                nx3 = x3 - cx3
                ny3 = y3
                nz3 = z3 - cz3
                length3 = np.sqrt(nx3*nx3 + ny3*ny3 + nz3*nz3)
                nx3, ny3, nz3 = nx3/length3, ny3/length3, nz3/length3
                
                # Normal 4
                nx4 = x4 - cx4
                ny4 = y4
                nz4 = z4 - cz4
                length4 = np.sqrt(nx4*nx4 + ny4*ny4 + nz4*nz4)
                nx4, ny4, nz4 = nx4/length4, ny4/length4, nz4/length4
                
                # First triangle (1-2-3)
                vertices.extend([x1, y1, z1, nx1, ny1, nz1])
                vertices.extend([x2, y2, z2, nx2, ny2, nz2])
                vertices.extend([x3, y3, z3, nx3, ny3, nz3])
                
                # Second triangle (1-3-4)
                vertices.extend([x1, y1, z1, nx1, ny1, nz1])
                vertices.extend([x3, y3, z3, nx3, ny3, nz3])
                vertices.extend([x4, y4, z4, nx4, ny4, nz4])
        
        return np.array(vertices, dtype='f4'), np.array([], dtype='i4')

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