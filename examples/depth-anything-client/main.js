"use client"
import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { pipeline, env, RawImage } from '@xenova/transformers';
// import { pipeline, env, RawImage } from '@huggingface/transformers';
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
env.backends.onnx.wasm.numThreads = 6;
// env.backends.onnx.wasm.wasmPaths = 'https://noahcohn.com/transformers/dist/';

const DEFAULT_SCALE = 0.223;
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');
status.textContent = 'Loading model...';

// const depth_estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf',{dtype:'f32',device:'webgpu'});
// const depth_estimator = await pipeline('depth-estimation', 'onnx-community/depth-anything-v2-small',{device:'webgpu'});
const upscaler = await pipeline('image-to-image', 'Xenova/swin2SR-classical-sr-x2-64', {
    // quantized: false, // Uncomment this line to use the quantized version
});

status.textContent = 'Ready';
const channel = new BroadcastChannel('imageChannel');
const loaderChannel = new BroadcastChannel('loaderChannel');

let dnce=document.querySelector('#dance').checked;

async function sr(imageDataURL) {
console.log('run sr');
// imageContainer.innerHTML = '';
const img = new Image();
img.src = imageDataURL;
img.onload = async () => {
const canvas2 = document.createElement('canvas');
canvas2.width = img.width;
canvas2.height = img.height;
const ctx = canvas2.getContext('2d',{alpha:true,antialias:true});
// ctx.imageSmoothingEnabled =false;
ctx.drawImage(img, 0, 0);
// origImageData = ctx.getImageData(0, 0, img.width, img.height);
const image = new RawImage(origImageData.data, img.width, img.height,4);
const output = await upscaler(image);
const srimage= new THREE.CanvasTexture(output);
// srimage.anisotropy=4;
const exportCanvas = document.createElement('canvas');
exportCanvas.width = output.image.width;
exportCanvas.height = output.image.height;
const ctx2 = exportCanvas.getContext('2d',{alpha:true,antialias:true});
ctx2.drawImage(output.image, 0, 0);
const imageData = exportCanvas.toDataURL('image/jpeg',1.0);
// const blob2 = new Blob([imageData.data], { type: 'image/jpeg' });
const link2 = document.createElement('a');
link2.href = imageData;
link2.download = document.querySelector('#saveName').innerHTML+'.jpg';
link2.click();
console.log('save file');
};
// output.save('upscaled.png');
}

channel.onmessage = async (event) => {
const { imageDataURL} = event.data;
sr(imageDataURL );
};

fileUpload.addEventListener('change', function (e) {
    const file = e.target.files[0];
    if (!file) {
        return;
    }
	console.log('get image');
    const reader = new FileReader();
    reader.onload = e2 => sr(e2.target.result);
    reader.readAsDataURL(file);
});
