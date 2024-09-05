"use client"
import './style.css';

import { pipeline, env, RawImage } from '@xenova/transformers';
// import { pipeline, env, RawImage } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.0-alpha.14';
import { FXAAShader } from 'three/addons/shaders/FXAAShader.js';
import { LoopSubdivision } from 'three-subdivide';

import vtkFullScreenRenderWindow from '@kitware/vtk.js/Rendering/Misc/FullScreenRenderWindow';
import vtkActor from '@kitware/vtk.js/Rendering/Core/Actor';
import vtkMapper from '@kitware/vtk.js/Rendering/Core/Mapper';
import vtkImageData from '@kitware/vtk.js/Common/DataModel/ImageData';

env.allowLocalModels = false;
env.backends.onnx.wasm.proxy = true;
env.backends.onnx.wasm.numThreads = 4;
const DEFAULT_SCALE = 0.223;
const status = document.getElementById('status');
const fileUpload = document.getElementById('upload');
const imageContainer = document.getElementById('container');
const example = document.getElementById('example');
status.textContent = 'Loading model...';
const depth_estimator = await pipeline('depth-estimation', 'Xenova/depth-anything-small-hf',{device:'webgpu'});
status.textContent = 'Ready';
const channel = new BroadcastChannel('imageChannel');
const loaderChannel = new BroadcastChannel('loaderChannel');

async function predict(imageDataURL) {
imageContainer.innerHTML = '';
const img =document.createElement('img');
img.src = imageDataURL;
img.onload = async () => {
const canvas2 = document.createElement('canvas');
canvas2.width = img.width;
canvas2.height = img.height;
const ctx = canvas2.getContext('2d',{alpha:true,antialias:true});
// ctx.imageSmoothingEnabled =false;
ctx.drawImage(img, 0, 0);
const image = new RawImage(imageData.data, img.width, img.height,4);

const { depth } = await depth_estimator(image);
    const vtkDepthData = vtkImageData.newInstance();
    const scalars = vtkDepthData.getPointData().getScalars();
    scalars.setData(depthData.dataSync(), 1); // Assuming single-channel depth data
    // Create vtkMapper and vtkActor
    const mapper = vtkMapper.newInstance();
    mapper.setInputData(vtkDepthData);
    const actor = vtkActor.newInstance();
    actor.setMapper(mapper);
    // Set up vtk.js renderer with the specified container
    const container = document.getElementById('vtk-container');
    const fullScreenRenderWindow = vtkFullScreenRenderWindow.newInstance({
        rootContainer: container,
        containerStyle: { height: '100%', width: '100%', position: 'relative' },
    });
    const renderer = fullScreenRenderWindow.getRenderer();
    const renderWindow = fullScreenRenderWindow.getRenderWindow();
    renderer.addActor(actor);
    renderer.resetCamera();
    renderWindow.render();

status.textContent = 'Analysing...';
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

loaderChannel.onmessage = async (event) => {
const { glbLocation } = event.data;
loadGLTFScene(glbLocation);
};

channel.onmessage = async (event) => {
const { imageDataURL} = event.data;
predict(imageDataURL );
};

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
