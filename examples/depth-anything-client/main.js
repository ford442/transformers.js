import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { pipeline, env, RawImage } from '@xenova/transformers';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;
const DEFAULT_SCALE = 0.25;
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');
status.textContent = 'Loading model...';
const depth_estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf',{backend: 'webgpu'});
status.textContent = 'Ready';
const channel = new BroadcastChannel('imageChannel');
const loaderChannel = new BroadcastChannel('loaderChannel');
let onSliderChange;
let scene,sceneL,rendererL,cameraL,loadCanvas,controlsL;
let depthE,materialE,clock;
let moveForward=false;
let moveBackward=false;
let moveLeft=false;
let moveRight=false;
let canJump=false;
let prevTime = performance.now();
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let yawObject, pitchObject;
let lightingUniformsGroup, lightCenters;
const container = document.getElementById( 'container' );
const pointLightsMax = 300;
const api = {
count: 200,
};

const vertexShader=`
uniform ViewData {
mat4 projectionMatrix;
mat4 viewMatrix;
};
uniform mat4 modelMatrix;
uniform mat3 normalMatrix;
in vec3 position;
in vec3 normal;
in vec2 uv;
out vec2 vUv;
out vec3 vPositionEye;
out vec3 vNormalEye;
void main(){
vec4 vertexPositionEye = viewMatrix * modelMatrix * vec4( position, 1.0 );
vPositionEye = (modelMatrix * vec4( position, 1.0 )).xyz;
vNormalEye = (vec4(normal , 1.)).xyz;
vUv = uv;
gl_Position = projectionMatrix * vertexPositionEye;
}
`;

const fragmentShader=`

precision highp float;
precision highp int;
precision highp sampler2D;
uniform sampler2D map;

uniform LightingData{
vec4 lightPosition[POINTLIGHTS_MAX];
vec4 lightColor[POINTLIGHTS_MAX];
float pointLightsCount;
};

#include <common>

float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
if ( cutoffDistance > 0.0 ) {
distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
}
return distanceFalloff;
}

in vec2 vUv;
in vec3 vPositionEye;
in vec3 vNormalEye;
out vec4 fragColor;
   
void main(){
vec4 texColor = texture(map, vUv);
vec3 finalColor = vec3(0.0); // Initialize final color
for (int x = 0; x < int(pointLightsCount); x++) {
vec3 offset = lightPosition[x].xyz - vPositionEye;
vec3 dirToLight = normalize(offset);
float distance = length(offset);
float diffuse = max(0.0, dot(vNormalEye, dirToLight));
float attenuation = 1.0 / (distance * distance);
vec3 lightWeighting = lightColor[x].xyz * getDistanceAttenuation(distance, 4.0, 0.7);
finalColor += texColor.rgb * diffuse * attenuation * lightWeighting; 
}
fragColor = vec4(finalColor, 1.0); 
}
`;

async function predict(imageDataURL) {
imageContainer.innerHTML = '';
const img = new Image();
img.src = imageDataURL;
img.onload = async () => {
const canvas2 = document.createElement('canvas');
canvas2.width = img.width;
canvas2.height = img.height;
const ctx = canvas2.getContext('2d',{alpha:true});
ctx.drawImage(img, 0, 0);
const imageData = ctx.getImageData(0, 0, img.width, img.height);
const image = new RawImage(imageData.data, img.width, img.height,4);
const { canvas, setDisplacementMap } = setupScene(imageDataURL, image.width, image.height);
imageContainer.append(canvas);
const { depth } = await depth_estimator(image);
status.textContent = 'Analysing...';
setDisplacementMap(depth.toCanvas());
depthE=depth;
status.textContent = '';
const slider = document.createElement('input');
slider.type = 'range';
slider.min = 0;
slider.max = 1;
slider.step = 0.01;
slider.addEventListener('input', (e) => {
onSliderChange(parseFloat(e.target.value));
});
slider.defaultValue = DEFAULT_SCALE;
imageContainer.append(slider);
};
}

function setupScene(imageDataURL, w, h) {
const canvas = document.createElement('canvas');
const width = canvas.width = imageContainer.offsetWidth;
const height = canvas.height = imageContainer.offsetHeight;
scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(30, width / height, 0.01, 10);
camera.position.z = 2;
scene.add(camera);
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setSize(width, height);
renderer.setPixelRatio(window.devicePixelRatio);
const light = new THREE.AmbientLight(0xffffff, 2);
scene.add(light);
const image = new THREE.TextureLoader().load(imageDataURL);
image.colorSpace = THREE.SRGBColorSpace;
lightingUniformsGroup = new THREE.UniformsGroup();
lightingUniformsGroup.setName( 'LightingData' );
const data = [];
const dataColors = [];
lightCenters = [];
for ( let i = 0; i < pointLightsMax; i ++ ) {
const col = new THREE.Color( 0xffffff * Math.random() ).toArray();
const x = Math.random() * 50 - 25;
const z = Math.random() * 50 - 25;
data.push( new THREE.Uniform( new THREE.Vector4( x, 1, z, 0 ) ) ); // light position
dataColors.push( new THREE.Uniform( new THREE.Vector4( col[ 0 ], col[ 1 ], col[ 2 ], 0 ) ) ); // light color
lightCenters.push( { x, z } );
}
lightingUniformsGroup.add( data ); // light position
lightingUniformsGroup.add( dataColors ); // light position
lightingUniformsGroup.add( new THREE.Uniform( pointLightsMax ) ); // light position
const cameraUniformsGroup = new THREE.UniformsGroup();
cameraUniformsGroup.setName( 'ViewData' );
cameraUniformsGroup.add( new THREE.Uniform( camera.projectionMatrix ) ); // projection matrix
cameraUniformsGroup.add( new THREE.Uniform( camera.matrixWorldInverse ) ); // view matrix
const material = new THREE.RawShaderMaterial( {
uniforms: {
map:{value:image},
modelMatrix: { value: null },
normalMatrix: { value: null }
},
uniformsGroups: [ cameraUniformsGroup, lightingUniformsGroup ],
name: 'Box',
defines: {
POINTLIGHTS_MAX: pointLightsMax
},
vertexShader: vertexShader,
fragmentShader: fragmentShader,
glslVersion: THREE.GLSL3
} );
material.displacementScale = DEFAULT_SCALE;
const setDisplacementMap = (canvas) => {
material.displacementMap = new THREE.CanvasTexture(canvas);
materialE=material;
material.needsUpdate = true;
}
const setDisplacementScale = (scale) => {
material.displacementScale = scale;
material.needsUpdate = true;
}
onSliderChange = setDisplacementScale;
const [pw, ph] = w > h ? [1, h / w] : [w / h, 1];
const geometry = new THREE.PlaneGeometry(pw, ph, w, h);
clock = new THREE.Clock();
const plane = new THREE.Mesh(geometry, material);
plane.material.uniformsGroups = [ cameraUniformsGroup, lightingUniformsGroup ];
plane.material.uniforms.modelMatrix.value = plane.matrixWorld;
plane.material.uniforms.normalMatrix.value = plane.normalMatrix;
plane.rotation.x = - Math.PI / 2;
plane.position.y = - 1;
scene.add(plane);
const gridSize = { x: 10, y: 1, z: 10 };
const spacing = 6;
for ( let i = 0; i < gridSize.x; i ++ ) {
for ( let j = 0; j < gridSize.y; j ++ ) {
for ( let k = 0; k < gridSize.z; k ++ ) {
const mesh = new THREE.Mesh( geometry, material.clone() );
mesh.name = 'Sphere';
mesh.material.uniformsGroups = [ cameraUniformsGroup, lightingUniformsGroup ];
mesh.material.uniforms.modelMatrix.value = mesh.matrixWorld;
mesh.material.uniforms.normalMatrix.value = mesh.normalMatrix;
scene.add( mesh );
mesh.position.x = i * spacing - ( gridSize.x * spacing ) / 2;
mesh.position.y = 0;
mesh.position.z = k * spacing - ( gridSize.z * spacing ) / 2;
}
}
}
material.uniforms.map = { value: image }; 

const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true;
lightingUniformsGroup.uniforms[ 2 ].value =200;
renderer.setAnimationLoop(() => {
  // Moving Lights
const elapsedTime = clock.getElapsedTime();
const lights = lightingUniformsGroup.uniforms[ 0 ];
const radius = 5;
const speed = 0.5;
for ( let i = 0; i < lights.length; i ++ ) {
const light = lights[ i ];
const center = lightCenters[ i ];
const angle = speed * elapsedTime + i * 0.5;
const x = center.x + Math.sin( angle ) * radius;
const z = center.z + Math.cos( angle ) * radius;
light.value.set( x, 1, z, 0 );
}
  //  //
renderer.render(scene, camera);
controls.update();
});
window.addEventListener('resize', () => {
const width = imageContainer.offsetWidth;
const height = imageContainer.offsetHeight;
camera.aspect = width / height;
camera.updateProjectionMatrix();
renderer.setSize(width, height);
}, false);
return {
canvas: renderer.domElement,
setDisplacementMap,
};
}

function loadGLTFScene(gltfFilePath) {
console.log('got file path:',gltfFilePath);
imageContainer.innerHTML = '';
const loader = new GLTFLoader();
const textureLoader = new THREE.TextureLoader();
loadCanvas = document.createElement('canvas');
loadCanvas.id='evi';
loadCanvas.style.position='absolute';
loadCanvas.style.zindex=2100;
loadCanvas.style.top=0;
const width = loadCanvas.width = window.innerHeight;
const height = loadCanvas.height = window.innerHeight;
sceneL = new THREE.Scene();
loader.load(document.querySelector('#saveName').innerHTML+'.glb', function (gltf) {
console.log('load scene');
sceneL.add(gltf.scene);
const planeL = gltf.scene.children.find(child => child.isMesh);
if (planeL) {
const material = planeL.material;
material.needsUpdate = true;
material.displacementScale = 0.35;
textureLoader.load(document.querySelector('#saveName').innerHTML+'.jpg', function(texture) {
material.displacementMap = texture;
material.needsUpdate = true;
});
} else {
console.warn("No mesh found in the glTF scene.");
}
sceneL.add(planeL);
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
sceneL.add(ambientLight);
cameraL = new THREE.PerspectiveCamera(30, width / height, 0.01, 10);
cameraL.position.z = 2;
sceneL.add(cameraL);
rendererL = new THREE.WebGLRenderer({ loadCanvas, antialias: true });
rendererL.setSize(width, height);
rendererL.setPixelRatio(window.devicePixelRatio);
rendererL.domElement.id='mvi';
rendererL.domElement.style.position='absolute';
rendererL.domElement.style.zindex=2950;
rendererL.domElement.style.top=0;
imageContainer.appendChild(loadCanvas);
imageContainer.appendChild( rendererL.domElement );
controlsL = new OrbitControls( cameraL, rendererL.domElement );
console.log('render');
animate();
}, undefined, function (error) {
console.error(error);
});
}

function animate() {
requestAnimationFrame( animate );
  // Object wobble
const time = performance.now() * 0.001; // Get time in seconds
const wobbleAmount = 0.05;
const wobbleSpeed = 2;
cameraL.position.x = wobbleAmount * Math.sin(time * wobbleSpeed);
cameraL.position.y = wobbleAmount * Math.cos(time * wobbleSpeed * 1.2);
cameraL.rotation.z = wobbleAmount * 0.5 * Math.sin(time * wobbleSpeed * 0.8);
cameraL.lookAt(sceneL.position);
rendererL.render( sceneL, cameraL );
controlsL.update();
}

loaderChannel.onmessage = async (event) => {
const { glbLocation } = event.data;
loadGLTFScene(glbLocation);
};

channel.onmessage = async (event) => {
const { imageDataURL} = event.data;
predict(imageDataURL );
};

async function saveSceneAsGLTF() {
const exporter = new GLTFExporter();
try {
const options = {
binary: true,
// embedImages: true, // Embed all textures, including the displacement map
};
const gltf = await exporter.parseAsync(scene, options);
const blob = new Blob([gltf], { type: 'application/octet-stream' });
const link = document.createElement('a');
link.href = URL.createObjectURL(blob);
link.download = document.querySelector('#saveName').innerHTML+'.glb'; // Use .glb extension for binary glTF
link.click();
const displacementMap = materialE.displacementMap;
const exportCanvas = document.createElement('canvas');
exportCanvas.width = displacementMap.image.width;
exportCanvas.height = displacementMap.image.height;
const ctx = exportCanvas.getContext('2d');
ctx.drawImage(displacementMap.image, 0, 0);
const imageData = exportCanvas.toDataURL('image/jpeg',1.0); ;
// const blob2 = new Blob([imageData.data], { type: 'image/jpeg' });
const link2 = document.createElement('a');
link2.href = imageData;
link2.download = document.querySelector('#saveName').innerHTML+'.jpg';
link2.click();
} catch (error) {
console.error('Error exporting glTF:', error);
}
}

document.querySelector('#savegltf').addEventListener('click',function(){
saveSceneAsGLTF();
});
