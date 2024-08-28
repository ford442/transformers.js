import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { pipeline, env, RawImage } from '@xenova/transformers';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';

// Since we will download the model from the Hugging Face Hub, we can skip the local model check
env.allowLocalModels = false;
// Proxy the WASM backend to prevent the UI from freezing
env.backends.onnx.wasm.proxy = true;
// Constants
const DEFAULT_SCALE = 0.25;

// Reference the elements that we will need
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');

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
// Create plane and rescale it so that max(w, h) = 1
const [pw, ph] = w > h ? [1, h / w] : [w / h, 1];
const geometry = new THREE.PlaneGeometry(pw, ph, w, h);
const plane = new THREE.Mesh(geometry, material);
scene.add(plane);
// Add orbit controls
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
controlsL = new PointerLockControls(cameraL,rendererL.domElement);

sceneL.add( controlsL.getObject() );
yawObject = controlsL.getObject();
pitchObject = cameraL; // Assuming the camera is the first child of yawObject
 
controlsL.addEventListener('lock', function () {
rendererL.setAnimationLoop(animate);
    // Add event listeners for mouse movement when Pointer Lock is activated
document.addEventListener('mousemove', onMouseMove, false);
});

controlsL.addEventListener('unlock', function () {
rendererL.setAnimationLoop(null);
    // Remove the mousemove event listener when Pointer Lock is deactivated
document.removeEventListener('mousemove', onMouseMove, false);
});
 
const onKeyDown = function ( event ) {
switch ( event.code ) {
case 'ArrowUp':
case 'KeyW':
moveForward = true;
break;
case 'ArrowLeft':
case 'KeyA':
moveLeft = true;
break;
case 'ArrowDown':
case 'KeyS':
moveBackward = true;
break;
case 'ArrowRight':
case 'KeyD':
moveRight = true;
break;
case 'Space':
if ( canJump === true ) velocity.y += 350;
canJump = false;
break;
}
};

const onKeyUp = function ( event ) {
switch ( event.code ) {
case 'ArrowUp':
case 'KeyW':
moveForward = false;
break;
case 'ArrowLeft':
case 'KeyA':
moveLeft = false;
break;
case 'ArrowDown':
case 'KeyS':
moveBackward = false;
break;
case 'ArrowRight':
case 'KeyD':
moveRight = false;
break;
}

};

document.addEventListener( 'keydown', onKeyDown );
document.addEventListener( 'keyup', onKeyUp );

document.querySelector('#controlBtn').addEventListener( 'click',function(){
controlsL.lock();
});

console.log('append canvas and render');
animate();
}, undefined, function (error) {
console.error(error);
});
 
}

function onMouseMove(event) {
  if (controlsL.isLocked === true) {
    const movementX = event.movementX || 0;
    const movementY = event.movementY || 0;

    yawObject.rotation.y -= movementX * 0.00001;
    pitchObject.rotation.x -= movementY * 0.00001;

    // Clamp the pitch rotation to prevent the camera from flipping upside down
    pitchObject.rotation.x = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitchObject.rotation.x));
  }
}

function animate() {
 requestAnimationFrame(animate);

  const time = performance.now();
  const delta = (time - prevTime) / 1000.0;

  velocity.x -= velocity.x * 10.0 * delta;
  velocity.z -= velocity.z * 10.0 * delta;

  // Prevent the camera from falling below a certain height (e.g., y = 0)
 // if (controlsL.getObject().position.y < 0) {
  //  velocity.y = 0;
 //   controlsL.getObject().position.y = 0;
 // } else {
  //  velocity.y -= 9.8 * delta; 
//  }

  direction.z = Number(moveForward) - Number(moveBackward);
  direction.x = Number(moveRight) - Number(moveLeft);

    // Get the camera's forward and right directions
    const forward = new THREE.Vector3(0, 0, -1);
    forward.applyQuaternion(cameraL.quaternion); 

    const right = new THREE.Vector3(1, 0, 0);
    right.applyQuaternion(cameraL.quaternion);

    // Calculate movement direction based on camera's orientation
// direction.copy(forward).multiplyScalar(Number(moveForward) - Number(moveBackward));
// direction.add(right).multiplyScalar(Number(moveRight) - Number(moveLeft));
direction.normalize(); 

if (moveForward || moveBackward || moveLeft || moveRight) {
    velocity.x -= direction.x * 5.0 * delta;
    velocity.z += direction.z * 5.0 * delta; // Keep the '+' here as it's now working correctly
  }
  // Directly update the camera's position based on velocity and delta time
  controlsL.getObject().position.x -= velocity.x * delta;
  controlsL.getObject().position.z -= velocity.z * delta;
  controlsL.getObject().position.y += velocity.y * delta;

  prevTime = time;

  rendererL.render(sceneL, cameraL);
 }

loaderChannel.onmessage = async (event) => {
const { glbLocation } = event.data;
loadGLTFScene(glbLocation);
};
