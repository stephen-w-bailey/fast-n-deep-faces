from OpenGL import GL

def createAndCompileShader(stype,source):
    shader=GL.glCreateShader(stype)
    GL.glShaderSource(shader,source)
    GL.glCompileShader(shader)

    result=GL.glGetShaderiv(shader,GL.GL_COMPILE_STATUS)
    
    if (result!=1):
        raise Exception("Couldn't compile shader\nShader compilation Log:\n"+str(GL.glGetShaderInfoLog(shader)))
    return shader

vertex_shader = None
fragment_shader = None
program = None

def initialize():

    global vertex_shader
    global fragment_shader
    global program

    vertex_shader=createAndCompileShader(GL.GL_VERTEX_SHADER,"""
    #version 330 core

    layout(location = 0) in vec3 vertexPosition;
    layout(location = 1) in vec3 vertexNormal;
    layout(location = 2) in vec3 vertexColor;
    layout(location = 3) in float vertexVisible;

    out vec3 normal;
    out vec3 c;
    out vec3 position;
    out float visible;

    uniform mat4 MVP;
    uniform vec2 cop;
    uniform mat3 r;

    void main(void)
    {
        position = vertexPosition;
        gl_Position = MVP * vec4(vertexPosition,1);
        normal = vertexNormal;
        c = vertexColor;
        visible = vertexVisible;
    }
    """)

    fragment_shader=createAndCompileShader(GL.GL_FRAGMENT_SHADER,"""
    #version 330 core

    in vec3 position;
    in vec3 normal;
    in vec3 c;
    in float visible;

    out vec4 color;

    uniform vec3 amb;
    uniform vec3 direct;
    uniform vec3 directDir;
    uniform vec3 camT;
    uniform float specAng;
    uniform float specMag;

    void main(void)
    {
        vec3 n = normalize(normal);
        vec3 l = -normalize(directDir);
        float ref = max(0, dot(n,l));
        vec3 d = direct * ref;
        vec3 tex = d * c;

        // Compute shininess
        vec3 refDir = (2*dot(n,l)*n-l);
        vec3 camDiff = camT - position;
        vec3 H = normalize((l+normalize(camDiff)));
        camDiff = normalize(camDiff);
        float refDiff = max(0,dot(refDir,H));
        vec3 shine = direct * specMag * pow(refDiff,specAng);

        color = vec4(amb + visible * (tex + shine), 1.0);
    }
    """)

    program = GL.glCreateProgram()
    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)
