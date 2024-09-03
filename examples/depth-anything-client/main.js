"use client"
import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { pipeline, env, RawImage } from '@xenova/transformers';
// import { pipeline, env, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.14';
import { PointerLockControls } from 'three/addons/controls/PointerLockControls.js';
import { FlyControls } from 'three/addons/controls/FlyControls.js';
import { FirstPersonControls } from 'three/addons/controls/FirstPersonControls.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from 'three/addons/postprocessing/ShaderPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { LoopSubdivision } from 'three-subdivide';

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;
env.backends.onnx.wasm.numThreads = 1;
const DEFAULT_SCALE = 0.384;
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');
status.textContent = 'Loading model...';
const depth_estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf',{device:'webgpu'});
status.textContent = 'Ready';
const channel = new BroadcastChannel('imageChannel');
const loaderChannel = new BroadcastChannel('loaderChannel');
let onSliderChange;
let scene,sceneL,rendererL,cameraL,loadCanvas,controlsL;
let depthE,materialE;
let composer1, composer2, fxaaPass;
let moveForward=false;
let moveBackward=false;
let moveLeft=false;
let moveRight=false;
let canJump=false;
let prevTime = performance.now();
const velocity = new THREE.Vector3();
const direction = new THREE.Vector3();
let yawObject, pitchObject; // Declare these variables at a higher scope
const clock= new THREE.Clock;

async function predict(imageDataURL) {
imageContainer.innerHTML = '';
const img = new Image();
img.src = imageDataURL;
img.onload = async () => {
const canvas2 = document.createElement('canvas');
canvas2.width = img.width;
canvas2.height = img.height;
const ctx = canvas2.getContext('2d',{alpha:true,antialias:true});
// ctx.imageSmoothingEnabled =false;
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
canvas.id='tvi';
const width = canvas.width = imageContainer.offsetWidth;
const height = canvas.height = imageContainer.offsetHeight;
scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(120, width / height, .01, 10000);
// const camera = new THREE.PerspectiveCamera(120, width / height);
camera.position.z = 1;
scene.add(camera);
// const renderer = new THREE.WebGPURenderer();
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true,premultipliedAlpha:false });
renderer.autoClear = false;
fxaaPass = new ShaderPass( FXAAShader );
const outputPass = new OutputPass();
const renderPass = new RenderPass();
composer1 = new EffectComposer( renderer );
composer1.addPass( renderPass );
composer1.addPass( outputPass );
const pixelRatio = renderer.getPixelRatio();
fxaaPass.material.uniforms[ 'resolution' ].value.x = 1 / ( container.offsetWidth * pixelRatio );
fxaaPass.material.uniforms[ 'resolution' ].value.y = 1 / ( container.offsetHeight * pixelRatio );
composer2 = new EffectComposer( renderer );
composer2.addPass( renderPass );
composer2.addPass( outputPass );
composer2.addPass( fxaaPass );
renderer.setSize(width, height);
renderer.setPixelRatio(window.devicePixelRatio);
const light = new THREE.AmbientLight(0xcc0000,.49305777);
scene.add(light);
const image = new THREE.TextureLoader().load(imageDataURL);
image.colorSpace = THREE.SRGBColorSpace;
const material = new THREE.MeshStandardMaterial({
map: image,
side: THREE.DoubleSide,
});
material.receiveShadow = true;
material.displacementScale = DEFAULT_SCALE;
const setDisplacementMap = (canvas) => {
material.displacementMap = new THREE.CanvasTexture(canvas);
material.roughness=.5;
// material.roughnessMap=image;
        //  bump map
const displacementMap = material.displacementMap;
const exportCanvas = document.createElement('canvas');
exportCanvas.width = displacementMap.image.width;
exportCanvas.height = displacementMap.image.height;
const ctx = exportCanvas.getContext('2d',{alpha:true,antialias:true});
ctx.drawImage(displacementMap.image, 0, 0);
// ctx.imageSmoothingEnabled=false;
const imageData = ctx.getImageData(0, 0, exportCanvas.width, exportCanvas.height);
const data = imageData.data;
	// get depthmap
const mesh = new THREE.Mesh(geometry, material);

	
// Invert the image data
for (let i = 0; i < data.length; i += 4) {
data[i] = 255 - data[i];     // Red
data[i + 1] = 255 - data[i + 1]; // Green
data[i + 2] = 255 - data[i + 2]; // Blue
// data[i + 3] is the alpha channel, leave it unchanged
}
// Put the inverted data back on the canvas
ctx.putImageData(imageData, 0, 0);
const imageDataUrl = exportCanvas.toDataURL('image/jpeg', 1.0);
const bumpTexture =new THREE.CanvasTexture(exportCanvas);
bumpTexture.colorSpace = THREE.SRGBColorSpace; // LinearSRGBColorSpace
material.bumpMap=bumpTexture;
material.bumpScale=1.25;
materialE=material;
material.needsUpdate = true;
}
const setDisplacementScale = (scale) => {
material.displacementScale = scale;
material.needsUpdate = true;
}
onSliderChange = setDisplacementScale;
const [pw, ph] = w > h ? [1, h / w] : [w / h, 1];
// const geometry = new THREE.PlaneGeometry(pw, ph, w*2, h*2);

// Add a displacement modifier
const params = {
    split:          true,       // optional, default: true
    uvSmooth:       false,      // optional, default: false
    preserveEdges:  false,      // optional, default: false
    flatOnly:       false,      // optional, default: false
    maxTriangles:   Infinity,   // optional, default: Infinity
};
const geometry = LoopSubdivision.modify(new THREE.PlaneGeometry(pw, ph, w*2, h*2), iterations, params);
	
const plane = new THREE.Mesh(geometry, material);
plane.receiveShadow = true;
plane.castShadow = true;
scene.add(plane);

	//  fog
const tfog = new THREE.Fog( 0xcccccc, 0.01, 105 );
scene.add(tfog);

      // Create Spotlights
const spotLight1 = new THREE.SpotLight(0x2217de, 34.420)
spotLight1.position.set(0, 1.38, 0.181)
spotLight1.castShadow = true;
spotLight1.angle = .15;
spotLight1.penumbra = 0.52;
spotLight1.decay = .02;
spotLight1.distance = 4.966776;
spotLight1.visible = true;
const frstSize = 80;
spotLight1.shadow.camera = new THREE.OrthographicCamera(-frstSize / 2,frstSize / 2,frstSize / 2,-frstSize / 2,1,80);
// Same position as LIGHT position.
spotLight1.shadow.camera.position.copy(spotLight1.position);
spotLight1.shadow.camera.lookAt(spotLight1.position);
scene.add(spotLight1.shadow.camera);
scene.add( spotLight1 );
spotLight1.target.position.set( 0, 0, 0 ); // Aim at the origin
scene.add( spotLight1.target ); 
const spotLight2 = new THREE.SpotLight(0xbd1300, 44.420234)
spotLight2.position.set(0, 2.38234, 0.81234)
spotLight2.castShadow = true;
spotLight2.angle = .2423232;
spotLight2.penumbra = 0.52223;
spotLight2.decay = .02;
spotLight2.distance = 4.778778;     
spotLight2.visible = true;
spotLight2.shadow.camera = new THREE.OrthographicCamera(-frstSize / 2,frstSize / 2,frstSize / 2,-frstSize / 2,1,80);
// Same position as LIGHT position.
spotLight2.shadow.camera.position.copy(spotLight2.position);
spotLight2.shadow.camera.lookAt(spotLight2.position);
scene.add(spotLight2.shadow.camera);
scene.add( spotLight2 );
spotLight2.target.position.set( 0, 0, 0 ); // Aim at the origin
scene.add( spotLight2.target );
const spotLight3 = new THREE.SpotLight(0xe7ff15, 39.420234)
spotLight3.position.set(0, 1.38234, 0.81234)
spotLight3.castShadow = true;
spotLight3.angle = .12423232;
spotLight3.penumbra = 0.52223;
spotLight3.decay = .02;
spotLight3.distance = 4.778778;     
spotLight3.visible = true;
spotLight3.shadow.camera = new THREE.OrthographicCamera(-frstSize / 2,frstSize / 2,frstSize / 2,-frstSize / 2,1,80);
// Same position as LIGHT position.
spotLight3.shadow.camera.position.copy(spotLight3.position);
spotLight3.shadow.camera.lookAt(spotLight3.position);
scene.add(spotLight3.shadow.camera);
scene.add( spotLight3 );
spotLight3.target.position.set( 0, 0, 0 ); // Aim at the origin
scene.add( spotLight3.target );

const spotLight4 = new THREE.SpotLight(0x09a80f, 29.420234)
spotLight4.position.set(1, 1.138234, 1.81234)
spotLight4.castShadow = true;
spotLight4.angle = .12423232;
spotLight4.penumbra = 0.52223;
spotLight4.decay = .02;
spotLight4.distance = 4.778778;     
spotLight4.visible = true;
spotLight4.shadow.camera = new THREE.OrthographicCamera(-frstSize / 2,frstSize / 2,frstSize / 2,-frstSize / 2,1,80);
// Same position as LIGHT position.
spotLight4.shadow.camera.position.copy(spotLight4.position);
spotLight4.shadow.camera.lookAt(spotLight4.position);
scene.add(spotLight4.shadow.camera);
scene.add( spotLight4 );
spotLight4.target.position.set( 0, 0, 0 ); // Aim at the origin
scene.add( spotLight4.target );
	
renderer.shadowMap.enabled = true;
renderer.shadowMap.needsUpdate = true;
renderer.shadowMap.toneMapping =THREE.CineonToneMapping;
      
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
// renderer.shadowMap.type = THREE.VSMShadowMap;

const controls = new OrbitControls( camera, renderer.domElement );
// controls.movementSpeed = 1; // Adjust as needed
// controls.lookSpeed  =145.2; 

const wobbleAmount = 0.07; // Increased amplitude for more pronounced movements
const wobbleSpeed = 5;     // Faster wobble speed
// Access the displacement map and its data
renderer.setAnimationLoop(() => {
const time = performance.now() * 0.001; 
        // Apply wobble to x and y positions
//	const randomOffset = 0.5-(Math.random() * 1.0); // Adjust 0.5 for randomness intensity
const wobbleAmount = 0.07;
const wobbleSpeed = 4;
	//  wobble
const maxWobbleX = 0.5; // Adjust as needed
const maxWobbleY = 0.3;

  plane.position.x = Math.min(Math.max(wobbleAmount * Math.sin(time * wobbleSpeed), -maxWobbleX), maxWobbleX);
  plane.position.y = Math.min(Math.max(wobbleAmount * Math.cos(time * 3.13 * 1.5), -maxWobbleY), maxWobbleY);

camera.position.x = Math.min(Math.max(wobbleAmount * Math.cos(time * wobbleSpeed), -maxWobbleX), maxWobbleX);
 camera.position.y = Math.min(Math.max(wobbleAmount * Math.sin(time * 3.13 * 1.5), -maxWobbleY), maxWobbleY);
// camera.position.z = wobbleAmount * 0.13 * Math.sin(time * wobbleSpeed * 0.777); // Add some z-axis movement
  // camera.rotation.z = wobbleAmount * 0.515 * Math.cos(time * wobbleSpeed * 0.778); 
// camera.lookAt(scene.position); // Make the camera look at the center

spotLight1.position.x *= Math.cos( time ) * .15;
spotLight1.position.z = Math.sin( time ) * 1.5;
spotLight2.position.x = Math.cos( time ) * 1.15;
spotLight2.position.z *= Math.sin( time ) * 1.25;
spotLight3.position.x = Math.cos( time ) *  1.15;
spotLight3.position.z = Math.sin( time ) *  .5;
spotLight4.position.x = Math.cos( time ) *  1.015;
spotLight4.position.z = Math.sin( time ) *  .665;
// lightHelper1.update();
// lightHelper2.update();
// controls.update( clock.getDelta() );
// controls.update();
renderer.render(scene, camera);
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
material.displacementBias=-0.15;
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
const wobbleAmount = 0.03; // Increased amplitude for more pronounced movements
const wobbleSpeed = 2;     // Faster wobble speed
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
const ctx = exportCanvas.getContext('2d',{alpha:true,antialias:true});
ctx.drawImage(displacementMap.image, 0, 0);
const imageData = exportCanvas.toDataURL('image/jpeg',1.0);
// const blob2 = new Blob([imageData.data], { type: 'image/jpeg' });
const link2 = document.createElement('a');
link2.href = imageData;
link2.download = document.querySelector('#saveName').innerHTML+'.jpg';
link2.click();
} catch (error) {
console.error('Error exporting glTF:', error);
}
}

fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }
    const reader = new FileReader();
    reader.onload = e2 => predict(e2.target.result);
    reader.readAsDataURL(file);
});

document.querySelector('#savegltf').addEventListener('click',function(){
saveSceneAsGLTF();
});
const lockBtn=document.querySelector('#lockButton');
lockBtn.addEventListener('click', () => {
document.querySelector('#tvi').requestPointerLock(); 
});
