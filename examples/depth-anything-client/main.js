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
let depthE,materialE;

let moveForward=false;
let moveBackward=false;
let moveLeft=false;
let moveRight=false;
let canJump=false;
let prevTime = performance.now();

const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let yawObject, pitchObject; // Declare these variables at a higher scope

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
const material = new THREE.MeshStandardMaterial({
map: image,
side: THREE.DoubleSide,
});
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
const plane = new THREE.Mesh(geometry, material);
scene.add(plane);
const controls = new OrbitControls( camera, renderer.domElement );
controls.enableDamping = true;
renderer.setAnimationLoop(() => {
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
   // Object dance - Faster and more energetic
const time = performance.now() * 0.001; 
const wobbleAmount = 0.07; // Increased amplitude for more pronounced movements
const wobbleSpeed = 4;     // Faster wobble speed
cameraL.position.x = wobbleAmount * Math.sin(time * wobbleSpeed);
cameraL.position.y = wobbleAmount * Math.cos(time * wobbleSpeed * 1.5); // More variation in y-axis frequency
cameraL.position.z = wobbleAmount * 0.3 * Math.sin(time * wobbleSpeed * 0.7); // Add some z-axis movement
cameraL.rotation.z = wobbleAmount * 0.5 * Math.sin(time * wobbleSpeed * 0.8); 

   /*  // Object wobble
const time = performance.now() * 0.001; // Get time in seconds
const wobbleAmount = 0.05; // Adjust the intensity of the wobble
const wobbleSpeed = 2; // Adjust the speed of the wobble
cameraL.position.x = wobbleAmount * Math.sin(time * wobbleSpeed);
cameraL.position.y = wobbleAmount * Math.cos(time * wobbleSpeed * 1.2); // Slightly different frequency for y
cameraL.rotation.z = wobbleAmount * 0.5 * Math.sin(time * wobbleSpeed * 0.8); // Add some rotation for more 3D effect
*/

cameraL.lookAt(sceneL.position); // Make the camera look at the center
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
