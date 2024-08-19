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
let depthE,materialE;

const displacementShaderMaterial = new THREE.ShaderMaterial({
    uniforms: {
        map: { value: null }, // The base color texture
        displacementMap: { value: null }, // The displacement map texture
        displacementScale: { value: 0.5 }, // Adjust the strength of the displacement
        // Add other uniforms as needed (e.g., for lighting)
    },
    vertexShader: `
        varying vec2 vUv;
        varying vec3 vNormal; // Varying for normal interpolation
        uniform sampler2D displacementMap;
        uniform float displacementScale;
        void main() {
            vUv = uv;
            vNormal = normalize( normalMatrix * normal ); // Calculate and interpolate normals
            // Sample the displacement map and apply it to the vertex position
            vec3 displacedPosition = position + normal * texture2D( displacementMap, vUv ).r * displacementScale;
            gl_Position = projectionMatrix * modelViewMatrix * vec4( displacedPosition, 1.0 );
        }
    `,
    fragmentShader: `
        varying vec2 vUv;
        varying vec3 vNormal; // Receive interpolated normal
        uniform sampler2D map;
        // Add other uniforms as needed (e.g., for lighting)
        void main() {
            // Basic lighting calculation (you can customize this)
            vec3 lightDir = normalize( vec3( 1.0, 1.0, 1.0 ) );
            float diffuse = clamp( dot( vNormal, lightDir ), 0.0, 1.0 );
            gl_FragColor = texture2D( map, vUv ) * diffuse;
        }
    `
});

function bakeDisplacement(mesh, displacementMap) {
  const geometry = mesh.geometry;
  const positionAttribute = geometry.attributes.position;
  const uvAttribute = geometry.attributes.uv;
if(displacementMap){
  for (let i = 0; i < positionAttribute.count; i++) {
    const uv = new THREE.Vector2(uvAttribute.getX(i), uvAttribute.getY(i));
    const displacement = displacementMap.getPixel(uv.x, uv.y).r; // Assuming grayscale displacement map
    const originalPosition = new THREE.Vector3();
    originalPosition.fromBufferAttribute(positionAttribute, i);
    const offset = mesh.geometry.attributes.normal.clone().multiplyScalar(displacement * material.displacementScale);
    const newPosition = originalPosition.add(offset);
    positionAttribute.setXYZ(i, newPosition.x, newPosition.y, newPosition.z);
  }
}
  geometry.attributes.position.needsUpdate = true;
  geometry.computeVertexNormals(); // Recalculate normals
}

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
const { depth } = await depth_estimator(image);
status.textContent = 'Analysing...';
setDisplacementMap(depth.toCanvas());
depthE=depth;
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
const textureLoader = new THREE.TextureLoader();
loadCanvas = document.createElement('canvas');
loadCanvas.id='evi';
loadCanvas.style.position='absolute';
loadCanvas.style.zindex=2100;
loadCanvas.style.top=0;
const width = loadCanvas.width = window.innerHeight;
const height = loadCanvas.height = window.innerHeight;
sceneL = new THREE.Scene();
loader.load('tiff2.glb', function (gltf) {
console.log('load scene');
sceneL.add(gltf.scene); 
const planeL = gltf.scene.children.find(child => child.isMesh);
if (planeL) {
const material = planeL.material;
material.needsUpdate = true;
material.displacementScale = 0.5;
var txtloc2='tiff2.glb.jpg';
textureLoader.load('tiff2.glb.jpg', function(texture) {
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
const { imageDataURL  } = event.data;
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
link.download = 'scene.glb'; // Use .glb extension for binary glTF
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
link2.download = 'displacementMap.jpg';
link2.click();
} catch (error) {
console.error('Error exporting glTF:', error);
// Handle the error appropriately (e.g., show a message to the user)
}
}

document.querySelector('#savegltf').addEventListener('click',function(){
saveSceneAsGLTF();
});
