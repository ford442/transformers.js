import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { pipeline, env, RawImage } from '@xenova/transformers';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;
// Proxy the WASM backend to prevent the UI from freezing
env.backends.onnx.wasm.proxy = true;
// Constants
const DEFAULT_SCALE = 0.75;

// Reference the elements that we will need
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');

// Create a new depth-estimation pipeline
status.textContent = 'Loading model...';
const depth_estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf');
status.textContent = 'Ready';

const channel = new BroadcastChannel('imageChannel');
const loaderChannel = new BroadcastChannel('loaderChannel');

let onSliderChange;
let scene,sceneL,rendererL,cameraL,loadCanvas,controlsL;

// Predict depth map for the given image
async function predict(imageDataURL) {
imageContainer.innerHTML = '';
// Load the image from the data URL
const img = new Image();
img.src = imageDataURL;
img.onload = async () => {
const canvas2 = document.createElement('canvas');
canvas2.width = img.width;
canvas2.height = img.height;
const ctx = canvas2.getContext('2d',{alpha:true});
ctx.drawImage(img, 0, 0);

// Get the image data from the canvas
const imageData = ctx.getImageData(0, 0, img.width, img.height);

// Create a RawImage from the imageData
const image = new RawImage(imageData.data, img.width, img.height,4);

// Set up scene and slider controls
const { canvas, setDisplacementMap } = setupScene(imageDataURL, image.width, image.height);
imageContainer.append(canvas);
status.textContent = 'Analysing...';
const { depth } = await depth_estimator(image);
setDisplacementMap(depth.toCanvas());
status.textContent = '';
 // Add slider control
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
material.needsUpdate = true;
}
const setDisplacementScale = (scale) => {
material.displacementScale = scale;
material.needsUpdate = true;
}
onSliderChange = setDisplacementScale;
// Create plane and rescale it so that max(w, h) = 1
const [pw, ph] = w > h ? [1, h / w] : [w / h, 1];
const geometry = new THREE.PlaneGeometry(pw, ph, w, h);
const plane = new THREE.Mesh(geometry, material);
scene.add(plane);
// Add orbit controls
const controls = new OrbitControls(camera, renderer.domElement);
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
imageContainer.innerHTML = '';
const loader = new GLTFLoader();
loadCanvas = document.createElement('canvas');
loadCanvas.id='lcanvas';
const width = loadCanvas.width = imageContainer.offsetWidth;
const height = loadCanvas.height = imageContainer.offsetHeight;
sceneL = new THREE.Scene();
loader.load('./scene.glb', function (gltf) {
console.log('load scene');
sceneL.add(gltf.scene); 
const planeL = gltf.scene.children.find(child => child.isMesh);
if (planeL) {
const material = planeL.material;
material.needsUpdate = true;
material.displacementScale = 0.5; 
// You might still need to set the displacementMap here if it's not embedded in the glTF
// For example, if you have a displacement map texture loaded elsewhere:
// material.displacementMap = yourDisplacementMapTexture;
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
imageContainer.appendChild(loadCanvas);
imageContainer.appendChild( rendererL.domElement );
controlsL = new OrbitControls( cameraL, rendererL.domElement );
console.log('append canvas and render');
animate();
}, undefined, function (error) {
console.error(error);
});
 
}

function animate() {
requestAnimationFrame( animate );
rendererL.render( sceneL, cameraL );
controlsL.update();
}

loaderChannel.onmessage = async (event) => {
const { glbLocation } = event.data;
loadGLTFScene(glbLocation);
};

channel.onmessage = async (event) => {
const { imageDataURL } = event.data;
predict(imageDataURL);
};

async function saveSceneAsGLTF() {
const exporter = new GLTFExporter();
try {
const options = {
binary: true,
embedImages: true // Embed all textures, including the displacement map
};
const gltf = await exporter.parseAsync(scene, options);
const blob = new Blob([gltf], { type: 'application/octet-stream' });
const link = document.createElement('a');
link.href = URL.createObjectURL(blob);
link.download = 'scene.glb'; // Use .glb extension for binary glTF
link.click();
} catch (error) {
console.error('Error exporting glTF:', error);
// Handle the error appropriately (e.g., show a message to the user)
}
}

document.querySelector('#savegltf').addEventListener('click',function(){
saveSceneAsGLTF();
});
