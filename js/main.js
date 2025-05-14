//import './style.css'
//import *  as THREE from 'three'


/*
const fr = new FileReader();
fr.onload = () => {
  splatData = new Uint8Array(fr.result);
  console.log(splatData)
  console.log("Loaded", Math.floor(splatData.length / rowLength));
  worker.postMessage({
    buffer: splatData.buffer,
    vertexCount: Math.floor(splatData.length / rowLength),
  });
}


fr.readAsArrayBuffer("./model.splat");
*/

let cameraParams = {
    id: 0,
    img_name: "00001",
    width: 1959,
    height: 1090,
    position: [
        -3.0089893469241797, -0.11086489695181866, -3.7527640949141428,
    ],
    rotation: [
        [0.876134201218856, 0.06925962026449776, 0.47706599800804744],
        [-0.04747421839895102, 0.9972110940209488, -0.057586739349882114],
        [-0.4797239414934443, 0.027805376500959853, 0.8769787916452908],
    ],
    fy: 1164.6601287484507,
    fx: 1159.5880733038064,
};

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, (2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

function makePerspectiveMatrix(fov, aspect, near, far) {
  const top = near * Math.tan((fov * Math.PI) / 360); // vertical FOV in degrees
  const height = 2 * top;
  const width = aspect * height;
  const left = -0.5 * width;
  const right = left + width;
  const bottom = top - height;

  const x = (2 * near) / (right - left);
  const y = (2 * near) / (top - bottom);
  const a = (right + left) / (right - left);
  const b = (top + bottom) / (top - bottom);
  const c = -far / (far - near);          // Note: different from RH system
  const d = -far * near / (far - near);   // Note: different from RH system

  return new Float32Array([
    x,  0,  a,   0,
    0,  y,  b,   0,
    0,  0,  c,   d,
    0,  0,  1,   0    // <- +1 for left-handed system (was -1 in RH)
  ]);
}
  console.log(makePerspectiveMatrix(80, 1.333333333, 0.005, 10000))


function updateProjectionMatrix() {
  const near = this.near;

  // Calculate top extent of the frustum
  let top = near * Math.tan(0.5 * Math.PI / 180 * this.fov) / this.zoom;
  let height = 2 * top;
  let width = this.aspect * height;
  let left = -0.5 * width;

  const view = this.view;

  // If using a subregion of the camera (view offset)
  if (view !== null && view.enabled) {
    const fullWidth = view.fullWidth;
    const fullHeight = view.fullHeight;

    left += view.offsetX * width / fullWidth;
    top -= view.offsetY * height / fullHeight;
    width *= view.width / fullWidth;
    height *= view.height / fullHeight;//False
  }

  // Apply film offset if present
  const filmOffset = this.filmOffset;
  if (filmOffset !== 0) {
    left += near * filmOffset / this.getFilmWidth();            // False
  }

  // Build the projection matrix
  this.projectionMatrix.makePerspective(
    left,
    left + width,
    top,
    top - height,
    near,
    this.far,
    this.coordinateSystem // used in r140+ for XR coordinate support
  );

  // Inverse projection for shaders or post-processing
  this.projectionMatrixInverse.copy(this.projectionMatrix).invert();
}

function makePerspective(left, right, top, bottom, near, far, coordinateSystem = PerspectiveCamera.DEFAULT_COORDINATE_SYSTEM) {
  const te = this.elements;

  // Calculate horizontal and vertical scale
  const x = 2 * near / (right - left);
  const y = 2 * near / (top - bottom);

  // Calculate offsets (center of frustum)
  const a = (right + left) / (right - left);
  const b = (top + bottom) / (top - bottom);

  let c, d;

  if (coordinateSystem === PerspectiveCamera.DEFAULT_COORDINATE_SYSTEM) {
    // Standard right-handed perspective matrix
    c = -(far + near) / (far - near);
    d = -2 * far * near / (far - near);
  } else {
    if (coordinateSystem !== PerspectiveCamera.ALTERNATE_COORDINATE_SYSTEM) {
      throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: " + coordinateSystem);
    }
    // Alternate (e.g. WebXR left-handed) system
    c = -far / (far - near);
    d = -far * near / (far - near);
  }

  // Fill the matrix elements
  te[0] = x;     te[4] = 0;    te[8]  = a;   te[12] = 0;
  te[1] = 0;     te[5] = y;    te[9]  = b;   te[13] = 0;
  te[2] = 0;     te[6] = 0;    te[10] = c;   te[14] = d;
  te[3] = 0;     te[7] = 0;    te[11] = -1;  te[15] = 0;

  return this;
}

/*

function getProjectionMatrix(fx, fy, width, height) {
    const znear = 0.2;
    const zfar = 200;
    return [
        [(2 * fx) / width, 0, 0, 0],
        [0, -(2 * fy) / height, 0, 0],
        [0, 0, zfar / (zfar - znear), 1],
        [0, 0, -(zfar * znear) / (zfar - znear), 0],
    ].flat();
}

*/




    const rowLength = 3 * 4 + 3 * 4 + 4 + 4;
    let splatData = new Uint8Array();
    let vertexCount = 0;
    let indexBuffer = new Uint32Array();
    let splatMesh;
    const loaded = {value: false};

async function main(){
await fetch("./js/point_cloud.splat").then(res => res.arrayBuffer()).then(buffer => {
                splatData = new Uint8Array(buffer);
                vertexCount = Math.floor(splatData.length / rowLength);
                indexBuffer = new Uint32Array(vertexCount);
                console.log("Loaded", vertexCount, "splats");
                const texture = generateTexture(splatData.buffer, vertexCount);
                splatMesh = createSplatMesh(texture, vertexCount, indexBuffer);
                console.log(splatMesh)
                loaded.value = true;
            });


function floatToHalf(float) {
                const _floatView = new Float32Array(1);
                const _int32View = new Int32Array(_floatView.buffer);
                _floatView[0] = float;
                const f = _int32View[0];

                const sign = (f >> 31) & 0x0001;
                let exp = (f >> 23) & 0x00ff;
                let frac = f & 0x007fffff;

                let newExp;
                if (exp === 0) {
                    newExp = 0;
                } else if (exp < 113) {
                    newExp = 0;
                    frac |= 0x00800000;
                    frac >>= (113 - exp);
                    if (frac & 0x01000000) {
                        newExp = 1;
                        frac = 0;
                    }
                } else if (exp < 142) {
                    newExp = exp - 112;
                } else {
                    newExp = 31;
                    frac = 0;
                }

                return (sign << 15) | (newExp << 10) | (frac >> 13);
            }

            function packHalf2x16(x, y) {
                return (floatToHalf(x) | (floatToHalf(y) << 16)) >>> 0;
            }

            function generateTexture(buffer, vertexCount) {
                if (!buffer) return;
                const f_buffer = new Float32Array(buffer);
                const u_buffer = new Uint8Array(buffer);
                const texwidth = 2048; // Adjusted width
                const texheight = Math.ceil((2 * vertexCount) / texwidth);
                const texdata = new Uint32Array(texwidth * texheight * 4);
                const texdata_c = new Uint8Array(texdata.buffer);
                const texdata_f = new Float32Array(texdata.buffer);

                for (let i = 0; i < vertexCount; i++) {
                    // x, y, z
                    texdata_f[8 * i + 0] = f_buffer[8 * i + 0];
                    texdata_f[8 * i + 1] = f_buffer[8 * i + 1];
                    texdata_f[8 * i + 2] = f_buffer[8 * i + 2];

                    // r, g, b, a
                    texdata_c[4 * (8 * i + 7) + 0] = u_buffer[32 * i + 24 + 0];
                    texdata_c[4 * (8 * i + 7) + 1] = u_buffer[32 * i + 24 + 1];
                    texdata_c[4 * (8 * i + 7) + 2] = u_buffer[32 * i + 24 + 2];
                    texdata_c[4 * (8 * i + 7) + 3] = u_buffer[32 * i + 24 + 3];

                    // quaternions and scale to covariance
                    const scale = [
                        f_buffer[8 * i + 3 + 0],
                        f_buffer[8 * i + 3 + 1],
                        f_buffer[8 * i + 3 + 2],
                    ];
                    const rot = [
                        (u_buffer[32 * i + 28 + 0] - 128) / 128,
                        (u_buffer[32 * i + 28 + 1] - 128) / 128,
                        (u_buffer[32 * i + 28 + 2] - 128) / 128,
                        (u_buffer[32 * i + 28 + 3] - 128) / 128,
                    ];

                    const M = [
                        1.0 - 2.0 * (rot[2] * rot[2] + rot[3] * rot[3]),
                        2.0 * (rot[1] * rot[2] + rot[0] * rot[3]),
                        2.0 * (rot[1] * rot[3] - rot[0] * rot[2]),

                        2.0 * (rot[1] * rot[2] - rot[0] * rot[3]),
                        1.0 - 2.0 * (rot[1] * rot[1] + rot[3] * rot[3]),
                        2.0 * (rot[2] * rot[3] + rot[0] * rot[1]),

                        2.0 * (rot[1] * rot[3] + rot[0] * rot[2]),
                        2.0 * (rot[2] * rot[3] - rot[0] * rot[1]),
                        1.0 - 2.0 * (rot[1] * rot[1] + rot[2] * rot[2]),
                    ].map((k, i) => k * scale[Math.floor(i / 3)]);

                    const sigma = [
                        M[0] * M[0] + M[3] * M[3] + M[6] * M[6],
                        M[0] * M[1] + M[3] * M[4] + M[6] * M[7],
                        M[0] * M[2] + M[3] * M[5] + M[6] * M[8],
                        M[1] * M[1] + M[4] * M[4] + M[7] * M[7],
                        M[1] * M[2] + M[4] * M[5] + M[7] * M[8],
                        M[2] * M[2] + M[5] * M[5] + M[8] * M[8],
                    ];

                    texdata[8 * i + 4] = packHalf2x16(4 * sigma[0], 4 * sigma[1]);
                    texdata[8 * i + 5] = packHalf2x16(4 * sigma[2], 4 * sigma[3]);
                    texdata[8 * i + 6] = packHalf2x16(4 * sigma[4], 4 * sigma[5]);
                }
                console.log(texdata);
                const texture = new THREE.DataTexture(
                    texdata,
                    texwidth,
                    texheight,
                    THREE.RGBAIntegerFormat,
                    THREE.UnsignedIntType
                );
                texture.internalFormat = 'RGBA32UI';
                texture.needsUpdate = true;
                texture.wrapS = THREE.ClampToEdgeWrapping;
                texture.wrapT = THREE.ClampToEdgeWrapping;
                texture.minFilter = THREE.NearestFilter;
                texture.magFilter = THREE.NearestFilter;
                return texture;
            }


function createSplatMesh(texture, count, indexBuffer) {
                const geometry = new THREE.PlaneGeometry(4, 4);
                geometry.setAttribute('index', new THREE.InstancedBufferAttribute(indexBuffer, 1));
                
                console.log(geometry.attributes)
                const uniforms = {
                    u_texture: { value: undefined },
                    projection: { value: new THREE.Matrix4() },
                    view: { value: new THREE.Matrix4() },
                    focal: { value: new THREE.Vector2(cameraParams.fx, cameraParams.fy) },
                    viewport: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) }
                };


             const material = new THREE.ShaderMaterial({
                                uniforms: uniforms,
                                vertexShader: `
                                    precision highp float;
            precision highp int;

            uniform highp usampler2D u_texture;
            uniform mat4 projection, view;
            uniform vec2 focal;
            uniform vec2 viewport;

            attribute uint index;

            varying vec4 vColor;
            varying vec2 vPosition;

            void main () {
                uvec4 cen = texelFetch(u_texture, ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10), 0);
                vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1) ;
                vec4 pos2d = projection * cam;

                float clip = 1.2 * pos2d.w;
                if (pos2d.z < -clip || pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                uvec4 cov = texelFetch(u_texture, ivec2(((uint(index) & 0x3ffu) << 1) | 1u, uint(index) >> 10), 0);
                vec2 u1 = unpackHalf2x16(cov.x), u2 = unpackHalf2x16(cov.y), u3 = unpackHalf2x16(cov.z);
                mat3 Vrk = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y);

                mat3 J = mat3(
                    focal.x / cam.z, 0., -(focal.x * cam.x) / (cam.z * cam.z),
                    0., focal.y / cam.z, -(focal.y * cam.y) / (cam.z * cam.z),
                    0., 0., 0.
                );
/*
                 float DEG2RAD = acos(-1.0f) / 180.;
    float FOV = 40.*DEG2RAD;
    float n = 0.005;
    float f = 10000.;
    float w = viewport.x;
    float h = viewport.y;
    float aspect = w/h;
    
    float t = n * tan(FOV);
    float r = t * aspect;

    //float r = n * tan(FOV);
    //float t = r * aspect;
    
    

    J = mat3(0.5*w*n/r/cam.z, 0., -0.5*w*n/r/(cam.z*cam.z)*cam.x,
            0., 0.5*h*n/t/cam.z, -0.5*h*n/t/(cam.z*cam.z)*cam.y,
            0., 0., 0.
            );
                                    */



                mat3 T = transpose(mat3(view)) * J;
                mat3 cov2d = transpose(T) * Vrk * T;

                float mid = (cov2d[0][0] + cov2d[1][1]) / 2.0;
                float radius = length(vec2((cov2d[0][0] - cov2d[1][1]) / 2.0, cov2d[0][1]));
                float lambda1 = mid + radius, lambda2 = mid - radius;

                if(lambda2 < 0.0) return;
                vec2 diagonalVector = normalize(vec2(cov2d[0][1], lambda1 - cov2d[0][0]));
                vec2 majorAxis = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
                vec2 minorAxis = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

                vColor = clamp(pos2d.z/pos2d.w+1.0, 0.0, 1.0) * vec4((cov.w) & 0xffu, (cov.w >> 8) & 0xffu, (cov.w >> 16) & 0xffu, (cov.w >> 24) & 0xffu) / 255.0;
                vPosition = position.xy;

                vec2 vCenter = vec2(pos2d) / pos2d.w;
                gl_Position = vec4(vCenter + (position.y * majorAxis + position.x * minorAxis) / viewport, 0.0, 1.0);


                    //gl_Position = vec4(vCenter + position.xy * minorAxis / viewport, 0.0, 1.0);
                }
                                `,
                    fragmentShader: `
                        precision highp float;

                        varying vec4 vColor;
                        varying vec2 vPosition;

                        void main () {
                            float A = -dot(vPosition, vPosition);
                            if (A < -4.0) discard;
                            float B = exp(A) * vColor.a;
                            gl_FragColor = vec4(B * vColor.rgb, B);
                        }
                    `,
                    transparent: true,
                    blending: THREE.CustomBlending,
                      blendSrc: THREE.OneMinusDstAlphaFactor,
                      blendDst: THREE.OneFactor,
                      blendSrcAlpha: THREE.OneMinusDstAlphaFactor,
                      blendDstAlpha: THREE.OneFactor,
                      blendEquation: THREE.AddEquation,
                      blendEquationAlpha: THREE.AddEquation,                  
                    depthTest: false,
                    depthWrite: false,
                });




                material.uniforms.u_texture.value = texture;
                material.uniforms.viewport.value = new THREE.Vector2(window.innerWidth, window.innerHeight)


                console.log(material.uniforms)
                const mesh = new THREE.InstancedMesh(geometry, material, count);
                mesh.frustumCulled = false;
                mesh.renderOrder = 1;
                //scene.sortObjects = true;
                //scene.add(mesh);
                //mesh.instanceMatrix.needsUpdate = true;

                return mesh;
            }


}
//main();
function updateSplatSorting(mesh, camera, vertexCount, splatData, indexBuffer) {
                if (!splatData.buffer) return;
                const f_buffer = new Float32Array(splatData.buffer);
                const viewProjMatrix = new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse);
                //console.log(viewProjMatrix)
                let maxDepth = -Infinity;
                let minDepth = Infinity;
                let sizeList = new Int32Array(vertexCount);
                //console.log(viewProjMatrix);
                for (let i = 0; i < vertexCount; i++) {
                    const depth =
                        ((viewProjMatrix.elements[2] * f_buffer[8 * i + 0] +
                            viewProjMatrix.elements[6] * f_buffer[8 * i + 1] +
                            viewProjMatrix.elements[10] * f_buffer[8 * i + 2]) *
                            4096) |
                        0;
                    sizeList[i] = depth;
                    if (depth > maxDepth) maxDepth = depth;
                    if (depth < minDepth) minDepth = depth;
                }
                const depthInv = (65535) / (maxDepth - minDepth);
                const counts0 = new Uint32Array(65536);
                for (let i = 0; i < vertexCount; i++) {
                    const depthIndex = ((sizeList[i] - minDepth) * depthInv) | 0;
                    sizeList[i] = depthIndex;
                    counts0[depthIndex]++;
                }
                const starts0 = new Uint32Array(65536);
                for (let i = 1; i < 65536; i++) {
                    starts0[i] = starts0[i - 1] + counts0[i - 1];
                }
                for (let i = 0; i < vertexCount; i++) {
                    indexBuffer[starts0[sizeList[i]]++] = i;
                }
                mesh.instanceMatrix.needsUpdate = true;
            }


AFRAME.registerComponent('custom-instanced-mesh', {
    init: function () {

    this.loaded = loaded;
    this.break = false;
      
    },

    tick: function(t, dt){
        if(loaded.value && ! this.break)
        {
            console.log("Finally loaded")
            this.el.object3D.add(splatMesh);
            this.break = true;
        }
        
    }
    //tick: function(t, dt){}
  });


//console.log(material.uniforms.u_texture.value);

AFRAME.registerComponent('camera-listener', {

  init: function () {

    this.loaded = loaded;

    const sceneEl = this.el.sceneEl;

/*
    this.el.sceneEl.renderer.setClearColor(0x000000, 0);
    const camera = this.el.sceneEl.camera;
    camera.projectionMatrix.set(...getProjectionMatrix(cameraParams.fx, cameraParams.fy, window.innerWidth, window.innerHeight));
    camera.projectionMatrix.transpose();
*/    
  },


  tick: function () {
    if(this.loaded.value)
    {


        const cameraEl = this.el.sceneEl.cameraEl;
        if(!cameraEl) return;
        const camera = cameraEl.getObject3D('camera');
        console.log('Camera Aspect Ratio:', camera.aspect);
        console.log('Camera Field of View:', camera.fov);
        console.log('Camera Near Clipping Plane:', camera.near);
        console.log('Camera Far Clipping Plane:', camera.far);
        console.log('Camera Position:', camera.position);
        console.log('Camera Rotation:', camera.rotation);
        console.log('Camera Matrix World:', camera.matrixWorld);
        /*
        var position = new THREE.Vector3();
    var quaternion = new THREE.Quaternion();
     const cameraEl = document.querySelector('[camera]');
    cameraEl.object3D.getWorldPosition(position);
      this.el.object3D.getWorldQuaternion(quaternion);
        console.log(position);
    console.log(quaternion);

        splatMesh.material.uniforms.view.value.copy(camera.matrixWorldInverse)
        splatMesh.material.uniforms.projection.value.copy(camera.projectionMatrix);
        updateSplatSorting(splatMesh, camera, vertexCount, splatData, indexBuffer);
        */
    }
    
  }
});

const dimensions = {h:null, w: null}




AFRAME.registerComponent('get-source-dimentions', {
    init: function(){ this.data = undefined},

    tick: function(){
        if (!this.data) {
              const arSystem = this.el.sceneEl.systems["arjs"];
              //console.log(arSystem)
              if (arSystem) {
                this.data = arSystem._arSession.arSource.domElement
                console.log("arToolkitSource ready!");
              } else {
                return; // still waiting
              }
            }

        //console.log("Actual source dimensions:", this.data.clientWidth, this.data.clientHeight);
            dimensions.h = this.data.clientHeight
            dimensions.w = this.data.clientWidth
    }
})

const marker = document.querySelector("a-marker");
const camera = document.getElementById("markerCam")

const globCam = {projectionMatrix: new THREE.Matrix4(), matrixWorldInverse: new THREE.Matrix4()}

AFRAME.registerComponent('get-matrix', {

    init:function(){
        this.arToolkitSource = window.arToolkitSource;
        this.marker = marker;
        //this.camera = camera.components['camera'].camera;
        this.once={val1:false}
        this.loaded = loaded;
    },

    tick: function(){
        if(!this.loaded.value) return;

            //console.log(relativeMatrix)
            splatMesh.material.uniforms.view.value.copy(this.marker.object3D.matrixWorld)
            globCam.matrixWorldInverse = this.marker.object3D.matrixWorld.clone()
            updateSplatSorting(splatMesh, globCam, vertexCount, splatData, indexBuffer);

        //console.log("Source Dimentions: ", this.arToolkitSource.domElement.clientWidth, this.arToolkitSource.domElement.clientHeight);
    }


}
)

AFRAME.registerComponent('log-camera-params', {
  init: function () {
    this.sceneEl = this.el.sceneEl;
    this.loaded = loaded;
    this.oldViewMat = new THREE.Matrix4();

    this.once={val1:false}
 

    console.log("up to here correct")
    this.sceneEl.addEventListener('loaded', () => {
      const markerCamera = camera
      if (markerCamera) {
        this.camera = markerCamera.components['camera'].camera;
        if (this.camera) {
            this.sceneEl.renderer.setClearColor(0x000000, 0);
            
            
            
            

          //console.log('Camera Projection Matrix:', this.camera.projectionMatrix.toArray());
          //console.log('Camera Position:', this.camera.position.toArray());
          // Access other camera properties as needed: fov, aspect, near, far, etc.
        } else {
          console.warn('Camera object not found on a-marker-camera.');
        }
      } else {
        console.warn('a-marker-camera entity not found.');
      }
    });
  },

  tick: function()
  {
    if(!this.loaded.value) return;
    if(!this.once.val1 && this.camera.aspect != Infinity)
    {
        /*
        this.camera.aspect = window.innerWidth/window.innerHeight;
        console.log(this.camera.aspect + " :)");
        */
        const v = new THREE.Vector2();
        this.sceneEl.renderer.getSize(v)
        console.log( v, ":=9876");
            console.log(this.sceneEl.renderer);
        const fovRad = THREE.MathUtils.degToRad(this.camera.fov);
        this.fy = window.innerHeight / (2 * Math.tan(fovRad / 2));
        this.fx = this.fy * this.camera.aspect;

        console.log(this.fx, this.fy)
        console.log(innerWidth, innerHeight)
        console.log(this.camera);
/*
        window.innerHeight / (2  * tan(FOV)) * aspect

        window.innerWidth * n / (n*tan(FOV)*aspect)
        float FOV = 40.*DEG2RAD;
        float n = 0.2;
        float f = 200.;
        float aspect = 1.333333333333;
        
        float t = n * tan(FOV);
        float r = t * aspect;

        //float r = n * tan(FOV);
        //float t = r * aspect;
        
        float w = viewport.x;
        float h = viewport.y;

        J = mat3(0.5*w*n/r/cam.z, 0., -0.5*w*n/r/(cam.z*cam.z)*cam.x,
                0., 0.5*h*n/t/cam.z, -0.5*h*n/t/(cam.z*cam.z)*cam.y,
                0., 0., 0.
                );
*/

        this.once.val1 = true;
    }

        splatMesh.visible = true;
        const projMat =  this.camera.projectionMatrix;
        this.fx = projMat.elements[0] * dimensions.w / 2
        this.fy = projMat.elements[5] * dimensions.h / 2


        splatMesh.material.uniforms.focal.value = new THREE.Vector2(this.fx, this.fy);
        //splatMesh.material.uniforms.view.value.copy(this.camera.matrixWorldInverse)
        splatMesh.material.uniforms.viewport.value = new THREE.Vector2(dimensions.w, dimensions.h)

        this.sceneEl.canvas.width = dimensions.w; // Get acutal source dimentions <----------------------------------------------------------------
        this.sceneEl.canvas.height = dimensions.h;
        this.camera.updateProjectionMatrix();
         // See how to obtain the "Actual source dimentionsw" <-------------------------------
        //splatMesh.material.uniforms.projection.value.set(...getProjectionMatrix(this.fx, .fy, window.innerWidth, window.innerHeight));
        //splatMesh.material.uniforms.projection.value.transpose();
        
        splatMesh.material.uniforms.projection.value.copy(this.camera.projectionMatrix);
        globCam.projectionMatrix = this.camera.projectionMatrix.clone()
        //updateSplatSorting(splatMesh, this.camera, vertexCount, splatData, indexBuffer);

        //console.log(dimensions.w, dimensions.h)
        //console.log('Camera Projection Matrix:', this.camera.projectionMatrix.toArray());
        //console.log('Camera Position:', this.camera.matrixWorldInverse.toArray());
  }
});


main();

