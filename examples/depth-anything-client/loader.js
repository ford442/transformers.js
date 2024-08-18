import './style.css';

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera( 75, window.innerWidth / window.innerHeight, 0.1, 1000 );
const renderer = new THREE.WebGLRenderer();
renderer.setSize( window.innerWidth, window.innerHeight );
document.body.appendChild( renderer.domElement );
const controls = new OrbitControls( camera, renderer.domElement );
const loader = new GLTFLoader();

loader.load(
 './scene.glb', function ( gltf ) {
scene.add( gltf.scene );
const plane = gltf.scene.children.find(child => child.isMesh);
const material = plane.material;
      // Apply displacement settings to the material
material.needsUpdate = true;
material.displacementScale = 0.5;
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5); // Color, intensity
scene.add(ambientLight);
}

// Adjust camera position if needed
camera.position.z = 5;

animate();
}, undefined, function ( error ) {
console.error( error );
} );

function animate() {
requestAnimationFrame( animate );
controls.update();
renderer.render( scene, camera );
}
