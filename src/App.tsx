import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js';
import { RectAreaLightUniformsLib } from 'three/examples/jsm/lights/RectAreaLightUniformsLib.js';
import {
  ARCHETYPE_PRESETS,
  AnimationState,
  ArchetypeId,
  CameraPresetId,
  DorsalType,
  EnemyConfig,
  HelmStyle,
  LightingPresetId,
  MATERIAL_PRESETS,
  MaterialPresetId,
  OffhandType,
  PartInspectionInfo,
  WeaponType,
  calculateThreatScore,
} from './types/enemy';
import {
   createProceduralTextureSet,
  ProceduralTextureSet,
} from './three/proceduralTextures';
import {
  BuiltEnemyRig,
  buildProceduralEnemy,
  updateEnemyAnimation,
  updateEnemyMaterialsInPlace,
} from './three/enemyBuilder';
import { exportEnemyToGLB } from './three/exporters';
import { BottomControlDock, DockTabId } from './components/BottomControlDock';
import { ExportModal } from './components/ExportModal';

const CAMERA_PRESET_TARGETS: Record<
  CameraPresetId,
  { pos: [number, number, number]; target: [number, number, number] }
> = {
  hero_three_quarter: { pos: [2.85, 2.25, 3.65], target: [0, 1.35, 0] },
  front_ortho: { pos: [0, 1.65, 4.3], target: [0, 1.35, 0] },
  weapon_closeup: { pos: [-1.85, 1.45, 2.15], target: [-0.45, 1.2, 0.35] },
  head_portrait: { pos: [0.65, 2.35, 1.85], target: [0, 2.05, 0] },
  top_tactical: { pos: [0, 5.6, 2.2], target: [0, 1.0, 0] },
};

const LIGHTING_THEMES: Record<
  LightingPresetId,
  {
    bgHex: number;
    fogDensity: number;
    keyColor: number;
    keyIntensity: number;
    hemiSky: number;
    hemiGround: number;
    hemiIntensity: number;
    rimColor: number;
    rimIntensity: number;
  }
> = {
  dark_forge: {
    bgHex: 0x0b0d13,
    fogDensity: 0.042,
    keyColor: 0xffdca8,
    keyIntensity: 2.6,
    hemiSky: 0x38bdf8,
    hemiGround: 0x1e130c,
    hemiIntensity: 0.85,
    rimColor: 0xf59e0b,
    rimIntensity: 2.4,
  },
  studio_pbr: {
    bgHex: 0x11151f,
    fogDensity: 0.028,
    keyColor: 0xfff7ed,
    keyIntensity: 2.8,
    hemiSky: 0x94a3b8,
    hemiGround: 0x1e293b,
    hemiIntensity: 1.1,
    rimColor: 0x38bdf8,
    rimIntensity: 2.0,
  },
  frost_crypt: {
    bgHex: 0x09131c,
    fogDensity: 0.048,
    keyColor: 0xe0f2fe,
    keyIntensity: 2.3,
    hemiSky: 0x0284c7,
    hemiGround: 0x0f172a,
    hemiIntensity: 0.95,
    rimColor: 0x2dd4bf,
    rimIntensity: 2.6,
  },
  solar_arena: {
    bgHex: 0x18120e,
    fogDensity: 0.035,
    keyColor: 0xfef08a,
    keyIntensity: 3.1,
    hemiSky: 0xf59e0b,
    hemiGround: 0x271712,
    hemiIntensity: 1.0,
    rimColor: 0xfb7185,
    rimIntensity: 2.2,
  },
};

export default function App() {
  const [config, setConfig] = useState<EnemyConfig>(() =>
    structuredClone(ARCHETYPE_PRESETS.elder_dragon)
  );
  const [activeTab, setActiveTab] = useState<DockTabId>('catalog');
  const [isDockCollapsed, setIsDockCollapsed] = useState(false);
  const [explodedAmount, setExplodedAmount] = useState(0);
  const [animState, setAnimState] = useState<AnimationState>('idle');
  const [animSpeed, setAnimSpeed] = useState(1.0);
  const [cameraPreset, setCameraPreset] = useState<CameraPresetId>('hero_three_quarter');
  const [lightingPreset, setLightingPreset] = useState<LightingPresetId>('dark_forge');
  const [showHitboxes, setShowHitboxes] = useState(false);
  const [wireframe, setWireframe] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [isExportModalOpen, setIsExportModalOpen] = useState(false);
  const [exportStatusMessage, setExportStatusMessage] = useState<string | null>(null);
  const [inspectedPart, setInspectedPart] = useState<PartInspectionInfo | null>(null);
  const [webglLost, setWebglLost] = useState(false);
  const [meshMetrics, setMeshMetrics] = useState({
    totalVertices: 3240,
    totalTriangles: 2680,
    partCount: 12,
  });

  // Refs for isolated WebGL lifecycle (Zero canvas teardown on React state updates)
  const canvasMountRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const rigRef = useRef<BuiltEnemyRig | null>(null);
  const texturesRef = useRef<ProceduralTextureSet | null>(null);
  const isSoftwareGlRef = useRef<boolean>(false);
  const lightsRef = useRef<{
    keyLight: THREE.DirectionalLight;
    hemiLight: THREE.HemisphereLight;
    rimLight: THREE.SpotLight;
  } | null>(null);

  // Mutable animation & camera refs read inside requestAnimationFrame
  const animStateRef = useRef<AnimationState>(animState);
  animStateRef.current = animState;

  const animSpeedRef = useRef<number>(animSpeed);
  animSpeedRef.current = animSpeed;

  const explodedRef = useRef<number>(explodedAmount);
  explodedRef.current = explodedAmount;

  const autoRotateRef = useRef<boolean>(autoRotate);
  autoRotateRef.current = autoRotate;

  const camTransitionRef = useRef<{
    active: boolean;
    goalPos: THREE.Vector3;
    goalTarget: THREE.Vector3;
  }>({
    active: false,
    goalPos: new THREE.Vector3(2.85, 2.25, 3.65),
    goalTarget: new THREE.Vector3(0, 1.35, 0),
  });

  // Keyboard shortcut 'H' to toggle Zen Mode dock collapse
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement ||
        e.target instanceof HTMLSelectElement
      ) {
        return;
      }
      if (e.key.toLowerCase() === 'h') {
        setIsDockCollapsed((prev) => !prev);
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, []);

  // Initialize WebGLRenderer, RoomEnvironment, Pedestal, Lights & Loop once
  useEffect(() => {
    const mount = canvasMountRef.current;
    if (!mount) return;

    // 1. Software WebGL detection from GPU renderer string
    const isSoftwareGl = (() => {
      const probeCanvas = document.createElement('canvas');
      const gl = probeCanvas.getContext('webgl2') || probeCanvas.getContext('webgl');
      if (!gl) return true;
      const ext = gl.getExtension('WEBGL_debug_renderer_info');
      const param = ext ? ext.UNMASKED_RENDERER_WEBGL : gl.RENDERER;
      const gpu = String(gl.getParameter(param));
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      return /SwiftShader|llvmpipe|softpipe|Software|Basic Render/i.test(gpu);
    })();
    isSoftwareGlRef.current = isSoftwareGl;

    const width = mount.clientWidth || window.innerWidth;
    const height = mount.clientHeight || window.innerHeight;

    const renderer = new THREE.WebGLRenderer({
      antialias: !isSoftwareGl,
      powerPreference: 'high-performance',
    });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.12;
    renderer.setPixelRatio(isSoftwareGl ? 1 : Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = !isSoftwareGl;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    mount.innerHTML = '';
    mount.appendChild(renderer.domElement);

    // WebGL Context Loss & Restore listeners
    const onContextLost = (e: Event) => {
      e.preventDefault();
      setWebglLost(true);
    };
    const onContextRestored = () => {
      setWebglLost(false);
    };
    renderer.domElement.addEventListener('webglcontextlost', onContextLost);
    renderer.domElement.addEventListener('webglcontextrestored', onContextRestored);

    // 2. Scene & Studio RoomEnvironment Reflections
    const scene = new THREE.Scene();
    const initialTheme = LIGHTING_THEMES.dark_forge;
    scene.background = new THREE.Color(initialTheme.bgHex);
    scene.fog = new THREE.FogExp2(initialTheme.bgHex, initialTheme.fogDensity);
    sceneRef.current = scene;

    const pmremGenerator = new THREE.PMREMGenerator(renderer);
    scene.environment = pmremGenerator.fromScene(new RoomEnvironment(), 0.04).texture;
    pmremGenerator.dispose();

    // 3. Camera & OrbitControls
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 80);
    camera.position.set(2.85, 2.25, 3.65);
    cameraRef.current = camera;

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.35, 0);
    controls.enableDamping = !isSoftwareGl;
    controls.dampingFactor = 0.06;
    controls.minDistance = 1.3;
    controls.maxDistance = 11.5;
    controls.maxPolarAngle = Math.PI * 0.49;
    controls.update();
    controlsRef.current = controls;

    // Cancel scripted camera transition when user interacts manually
    const onControlsStart = () => {
      camTransitionRef.current.active = false;
    };
    controls.addEventListener('start', onControlsStart);

    // 4. Studio 3-Point Lighting Rig + Area Softboxes
    const keyLight = new THREE.DirectionalLight(initialTheme.keyColor, initialTheme.keyIntensity);
    keyLight.position.set(4.2, 6.8, 4.5);
    keyLight.castShadow = !isSoftwareGl;
    keyLight.shadow.mapSize.width = 512;
    keyLight.shadow.mapSize.height = 512;
    const d = 3.2;
    keyLight.shadow.camera.left = -d;
    keyLight.shadow.camera.right = d;
    keyLight.shadow.camera.top = d;
    keyLight.shadow.camera.bottom = -d;
    keyLight.shadow.camera.near = 0.5;
    keyLight.shadow.camera.far = 16;
    keyLight.shadow.bias = -0.0005;
    keyLight.shadow.normalBias = 0.02;
    scene.add(keyLight);

    const hemiLight = new THREE.HemisphereLight(
      initialTheme.hemiSky,
      initialTheme.hemiGround,
      initialTheme.hemiIntensity
    );
    hemiLight.position.set(0, 12, 0);
    scene.add(hemiLight);

    const rimLight = new THREE.SpotLight(
      initialTheme.rimColor,
      initialTheme.rimIntensity,
      18,
      Math.PI / 3,
      0.6,
      1.5
    );
    rimLight.position.set(-3.5, 4.2, -4.0);
    rimLight.lookAt(0, 1.4, 0);
    scene.add(rimLight);

    lightsRef.current = { keyLight, hemiLight, rimLight };

    if (!isSoftwareGl) {
      RectAreaLightUniformsLib.init();
      const softboxL = new THREE.RectAreaLight(0x38bdf8, 2.2, 2.4, 3.2);
      softboxL.position.set(-3.8, 2.4, 2.2);
      softboxL.lookAt(0, 1.3, 0);
      scene.add(softboxL);

      const softboxR = new THREE.RectAreaLight(0xf59e0b, 1.8, 2.4, 3.2);
      softboxR.position.set(3.8, 2.2, -1.8);
      softboxR.lookAt(0, 1.3, 0);
      scene.add(softboxR);
    }

    // 5. Procedural Textures & Grounded Tactical Forge Pedestal (y <= 0)
    const maxAnisotropy = renderer.capabilities.getMaxAnisotropy();
    const textures = createProceduralTextureSet(maxAnisotropy);
    texturesRef.current = textures;

    const stageGroup = new THREE.Group();
    stageGroup.name = 'Tactical_Forge_Stage';
    scene.add(stageGroup);

    // Beveled Obsidian Dais under y = 0
    const daisGeo = new THREE.CylinderGeometry(2.35, 2.65, 0.22, 16);
    const daisMat = new THREE.MeshStandardMaterial({
      color: 0x121620,
      metalness: 0.55,
      roughness: 0.38,
      bumpMap: textures.armorBumpMap,
      bumpScale: 0.01,
    });
    const daisMesh = new THREE.Mesh(daisGeo, daisMat);
    daisMesh.position.y = -0.11;
    daisMesh.receiveShadow = true;
    stageGroup.add(daisMesh);

    // Outer Bronze Trim Ring
    const trimRingGeo = new THREE.TorusGeometry(2.38, 0.035, 8, 48);
    trimRingGeo.rotateX(Math.PI / 2);
    const trimRingMat = new THREE.MeshStandardMaterial({
      color: 0x9a7b4f,
      metalness: 0.85,
      roughness: 0.25,
    });
    const trimRingMesh = new THREE.Mesh(trimRingGeo, trimRingMat);
    trimRingMesh.position.y = -0.01;
    stageGroup.add(trimRingMesh);

    // Runic Measurement Dial Decal (offset + polygonOffset to prevent z-fighting)
    const dialGeo = new THREE.PlaneGeometry(4.5, 4.5);
    dialGeo.rotateX(-Math.PI / 2);
    const dialMat = new THREE.MeshBasicMaterial({
      map: textures.pedestalRunesMap,
      transparent: true,
      opacity: 0.85,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -1,
      polygonOffsetUnits: -1,
    });
    const dialMesh = new THREE.Mesh(dialGeo, dialMat);
    dialMesh.position.y = 0.005;
    stageGroup.add(dialMesh);

    // Soft Radial Contact Shadow directly beneath the enemy at y = 0.012
    const shadowPlaneGeo = new THREE.PlaneGeometry(2.6, 2.6);
    shadowPlaneGeo.rotateX(-Math.PI / 2);
    const shadowPlaneMat = new THREE.MeshBasicMaterial({
      map: textures.contactShadowMap,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    });
    const shadowPlaneMesh = new THREE.Mesh(shadowPlaneGeo, shadowPlaneMat);
    shadowPlaneMesh.position.y = 0.012;
    stageGroup.add(shadowPlaneMesh);

    // Extended infinite floor plane in the fog
    const floorGeo = new THREE.PlaneGeometry(60, 60);
    floorGeo.rotateX(-Math.PI / 2);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x090b10,
      roughness: 0.88,
      metalness: 0.15,
    });
    const floorMesh = new THREE.Mesh(floorGeo, floorMat);
    floorMesh.position.y = -0.22;
    floorMesh.receiveShadow = true;
    stageGroup.add(floorMesh);

    // 6. Raycaster on Layer 1 for Interactive Part Inspection on Click
    const raycaster = new THREE.Raycaster();
    raycaster.layers.set(1);
    const pointer = new THREE.Vector2();
    let pointerDownPos = { x: 0, y: 0 };

    const onPointerDown = (e: PointerEvent) => {
      pointerDownPos = { x: e.clientX, y: e.clientY };
    };
    const onPointerUp = (e: PointerEvent) => {
      const dist = Math.hypot(e.clientX - pointerDownPos.x, e.clientY - pointerDownPos.y);
      if (dist > 6 || !rigRef.current) return;
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(rigRef.current.exportableGroup.children, true);
      if (hits.length > 0) {
        const info = hits[0].object.userData?.partInfo as PartInspectionInfo | undefined;
        if (info) {
          setInspectedPart(info);
        }
      }
    };
    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointerup', onPointerUp);

    // 7. Resize Handler
    const onResize = () => {
      if (!mount) return;
      const w = mount.clientWidth || window.innerWidth;
      const h = mount.clientHeight || window.innerHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };
    window.addEventListener('resize', onResize);

    // 8. Paced Render & Kinematic Loop (GPU rAF vs Software WebGL fenceSync afterDraw)
    const clock = new THREE.Clock();
    let frameId = 0;
    let isDisposed = false;

    function afterDraw(cb: () => void) {
      const gl = renderer.getContext() as WebGL2RenderingContext;
      if (!gl || typeof gl.fenceSync !== 'function') {
        frameId = window.setTimeout(cb, 80);
        return;
      }
      const sync = gl.fenceSync(gl.SYNC_GPU_COMMANDS_COMPLETE, 0);
      gl.flush();
      const poll = () => {
        if (isDisposed) return;
        const state = sync && gl.getSyncParameter(sync, gl.SYNC_STATUS);
        if (state === gl.UNSIGNALED) {
          frameId = window.setTimeout(poll, 5);
        } else {
          if (sync) gl.deleteSync(sync);
          frameId = window.setTimeout(cb, 80);
        }
      };
      frameId = window.setTimeout(poll, 5);
    }

    function tick() {
      if (isDisposed) return;
      const delta = Math.min(clock.getDelta(), 0.1);
      const elapsed = clock.getElapsedTime();

      // Smooth damped camera preset transitions
      if (camTransitionRef.current.active) {
        const gPos = camTransitionRef.current.goalPos;
        const gTar = camTransitionRef.current.goalTarget;
        camera.position.x = THREE.MathUtils.damp(camera.position.x, gPos.x, 6, delta);
        camera.position.y = THREE.MathUtils.damp(camera.position.y, gPos.y, 6, delta);
        camera.position.z = THREE.MathUtils.damp(camera.position.z, gPos.z, 6, delta);

        controls.target.x = THREE.MathUtils.damp(controls.target.x, gTar.x, 6, delta);
        controls.target.y = THREE.MathUtils.damp(controls.target.y, gTar.y, 6, delta);
        controls.target.z = THREE.MathUtils.damp(controls.target.z, gTar.z, 6, delta);

        if (
          camera.position.distanceToSquared(gPos) < 0.002 &&
          controls.target.distanceToSquared(gTar) < 0.002
        ) {
          camTransitionRef.current.active = false;
        }
      }

      if (autoRotateRef.current && !isSoftwareGl && rigRef.current) {
        rigRef.current.rootGroup.rotation.y += delta * 0.45;
      }

      if (rigRef.current) {
        updateEnemyAnimation(
          rigRef.current,
          animStateRef.current,
          elapsed,
          delta,
          animSpeedRef.current,
          explodedRef.current
        );
      }

      controls.update();
      renderer.render(scene, camera);

      if (isSoftwareGl) {
        afterDraw(tick);
      } else {
        frameId = requestAnimationFrame(tick);
      }
    }

    tick();

    return () => {
      isDisposed = true;
      cancelAnimationFrame(frameId);
      clearTimeout(frameId);
      window.removeEventListener('resize', onResize);
      controls.removeEventListener('start', onControlsStart);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointerup', onPointerUp);
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      renderer.domElement.removeEventListener('webglcontextrestored', onContextRestored);
      rigRef.current?.dispose();
      textures.dispose();
      daisGeo.dispose();
      daisMat.dispose();
      trimRingGeo.dispose();
      trimRingMat.dispose();
      dialGeo.dispose();
      dialMat.dispose();
      shadowPlaneGeo.dispose();
      shadowPlaneMat.dispose();
      floorGeo.dispose();
      floorMat.dispose();
      renderer.dispose();
    };
  }, []);

  // Structural & morphological signature: only rebuild geometry when anatomy or equipment changes
  const geometrySignature = JSON.stringify({
    archetype: config.archetype,
    bodyPlan: config.bodyPlan,
    helm: config.helm,
    weapon: config.weapon,
    offhand: config.offhand,
    dorsal: config.dorsal,
    morphology: config.morphology,
  });

  // Rebuild Procedural Enemy Mesh ONLY when structural/morphological signature changes
  useEffect(() => {
    const scene = sceneRef.current;
    const textures = texturesRef.current;
    if (!scene || !textures) return;

    const prevRotationY = rigRef.current ? rigRef.current.rootGroup.rotation.y : 0;
    if (rigRef.current) {
      scene.remove(rigRef.current.rootGroup);
      rigRef.current.dispose();
    }

    const newRig = buildProceduralEnemy(
      config,
      textures,
      isSoftwareGlRef.current,
      wireframe
    );
    newRig.rootGroup.rotation.y = prevRotationY;
    newRig.hitboxGroup.visible = showHitboxes;
    scene.add(newRig.rootGroup);
    rigRef.current = newRig;
    setMeshMetrics(newRig.metrics);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geometrySignature]);

  // Fast In-Place Update (< 0.1ms) for PBR Materials, Surface Textures, Colors, Wireframe & RPG Stats
  useEffect(() => {
    if (!rigRef.current || !texturesRef.current) return;
    updateEnemyMaterialsInPlace(
      rigRef.current,
      config,
      texturesRef.current,
      isSoftwareGlRef.current,
      wireframe
    );
  }, [config.materials, config.stats, config.name, config.title, wireframe]);

  // Toggle Hitbox & Socket helper visibility without rebuilding geometry
  useEffect(() => {
    if (rigRef.current) {
      rigRef.current.hitboxGroup.visible = showHitboxes;
    }
  }, [showHitboxes]);

  // Update Lighting Theme dynamically
  useEffect(() => {
    const scene = sceneRef.current;
    const lights = lightsRef.current;
    if (!scene || !lights) return;
    const theme = LIGHTING_THEMES[lightingPreset];
    scene.background = new THREE.Color(theme.bgHex);
    scene.fog = new THREE.FogExp2(theme.bgHex, theme.fogDensity);
    lights.keyLight.color.setHex(theme.keyColor);
    lights.keyLight.intensity = theme.keyIntensity;
    lights.hemiLight.color.setHex(theme.hemiSky);
    lights.hemiLight.groundColor.setHex(theme.hemiGround);
    lights.hemiLight.intensity = theme.hemiIntensity;
    lights.rimLight.color.setHex(theme.rimColor);
    lights.rimLight.intensity = theme.rimIntensity;
  }, [lightingPreset]);

  // Trigger smooth damped camera transition
  const handleCameraPresetChange = useCallback((preset: CameraPresetId) => {
    setCameraPreset(preset);
    const targetCfg = CAMERA_PRESET_TARGETS[preset];
    camTransitionRef.current.goalPos.set(...targetCfg.pos);
    camTransitionRef.current.goalTarget.set(...targetCfg.target);
    camTransitionRef.current.active = true;
  }, []);

  // Anatomically Coherent Procedural Randomizer (Zero missing fields or NaN values)
  const handleRandomize = useCallback(() => {
    const archetypes = Object.keys(ARCHETYPE_PRESETS) as ArchetypeId[];
    const matKeys = Object.keys(MATERIAL_PRESETS) as MaterialPresetId[];

    const pick = <T,>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
    const jitter = (base: number, min: number, max: number, spread = 0.22) => {
      const val = base * (1 + (Math.random() * 2 - 1) * spread);
      return Number(Math.max(min, Math.min(max, val)).toFixed(2));
    };

    const chosenArch = pick(archetypes);
    const basePreset = structuredClone(ARCHETYPE_PRESETS[chosenArch]);
    const bp = basePreset.bodyPlan;
    const bm = basePreset.morphology;

    // Pick anatomically compatible head/helm, dorsal, weapon & offhand for this bodyPlan
    const compatibleHelms: HelmStyle[] =
      bp === 'dragon_wyvern'
        ? ['draconic_snout', 'drake_fang_helm', 'avian_phoenix_beak']
        : bp === 'quadruped_beast'
        ? ['lupine_maw', 'tauren_bull_horns', 'avian_phoenix_beak', 'draconic_snout']
        : bp === 'arachnid_scorpion'
        ? ['arachnid_mandibles', 'crowned_skull', 'monolith_visor']
        : bp === 'serpentine_naga'
        ? ['draconic_snout', 'crowned_skull', 'valkyrie_crest', 'arachnid_mandibles']
        : [
            'horned_greathelm',
            'crowned_skull',
            'monolith_visor',
            'valkyrie_crest',
            'tauren_bull_horns',
            'drake_fang_helm',
          ];

    const compatibleDorsals: DorsalType[] =
      bp === 'dragon_wyvern'
        ? ['dragon_membrane_wings', 'seraph_wings', 'bio_chitin_spines', 'crystal_pylons']
        : bp === 'arachnid_scorpion'
        ? ['scorpion_stinger_tail', 'bio_chitin_spines', 'crystal_pylons']
        : bp === 'quadruped_beast'
        ? ['bio_chitin_spines', 'dragon_membrane_wings', 'seraph_wings', 'scorpion_stinger_tail', 'none']
        : [
            'tattered_mantle',
            'seraph_wings',
            'dragon_membrane_wings',
            'crystal_pylons',
            'runic_halo_ring',
            'bio_chitin_spines',
          ];

    const compatibleWeapons: WeaponType[] =
      bp === 'dragon_wyvern'
        ? ['dragon_breath_maw', 'feral_claws_fangs']
        : bp === 'quadruped_beast' || bp === 'arachnid_scorpion'
        ? ['feral_claws_fangs', 'venom_scythe_claws', 'twin_reaver_axes']
        : [
            'runic_greatsword',
            'obsidian_warhammer',
            'arcane_staff',
            'twin_reaver_axes',
            'radiant_halberd',
            'venom_scythe_claws',
          ];

    const compatibleOffhands: OffhandType[] =
      bp === 'dragon_wyvern' || bp === 'quadruped_beast' || bp === 'arachnid_scorpion'
        ? ['none', 'dual_blade']
        : ['tower_shield', 'runic_buckler', 'phylactery_orb', 'dual_blade', 'none'];

    // 65% chance to keep the archetype's signature material family or pick a dramatic variant
    const chosenMatKey = Math.random() > 0.35 ? basePreset.materials.preset : pick(matKeys);
    const matPreset = MATERIAL_PRESETS[chosenMatKey];

    setConfig((prev) => ({
      ...basePreset,
      id: `enemy_${chosenArch}_${Math.floor(100 + Math.random() * 899)}`,
      helm: Math.random() > 0.3 ? basePreset.helm : pick(compatibleHelms),
      weapon: Math.random() > 0.35 ? basePreset.weapon : pick(compatibleWeapons),
      offhand: Math.random() > 0.4 ? basePreset.offhand : pick(compatibleOffhands),
      dorsal: Math.random() > 0.35 ? basePreset.dorsal : pick(compatibleDorsals),
      morphology: {
        heightScale: jitter(bm.heightScale, 0.78, 1.42),
        torsoBulk: jitter(bm.torsoBulk, 0.76, 1.48),
        shoulderWidth: jitter(bm.shoulderWidth, 0.82, 1.48),
        hornLength: jitter(bm.hornLength, 0.18, 1.65),
        hornCurve: jitter(bm.hornCurve, 0.2, 1.35),
        pauldronSize:
          basePreset.raceCategory === 'starter_animals'
            ? 0.35
            : jitter(bm.pauldronSize, 0.65, 1.45),
        weaponScale: jitter(bm.weaponScale, 0.85, 1.45),
        dorsalScale: jitter(bm.dorsalScale, 0.65, 1.55),
        limbLength: jitter(bm.limbLength, 0.82, 1.28),
        neckLength: jitter(bm.neckLength, 0.25, 1.55),
        tailLength: jitter(bm.tailLength, 0.0, 1.65),
        snoutLength: jitter(bm.snoutLength, 0.35, 1.45),
        headCount:
          chosenArch === 'abyssal_hydra' || chosenArch === 'cerberus_hellhound'
            ? 3
            : basePreset.raceCategory === 'starter_animals'
            ? 1
            : Math.random() > 0.86
            ? 3
            : Math.random() > 0.78
            ? 2
            : 1,
        auraShardCount:
          basePreset.raceCategory === 'starter_animals'
            ? 0
            : Math.floor(Math.random() * 7) * 2 + 6,
        meshDensity: prev.morphology.meshDensity ?? 'medium',
      },
      materials: {
        preset: chosenMatKey,
        ...matPreset,
        texturePattern: prev.materials.texturePattern,
        textureRelief: prev.materials.textureRelief ?? 0.8,
      },
    }));
  }, []);

  // Binary .GLB Export Handler
  const handleExportGLB = useCallback(
    async (binary = true) => {
      if (!rigRef.current) return;
      try {
        setExportStatusMessage('Empaquetando geometría y materiales PBR...');
        const { sizeBytes, filename } = await exportEnemyToGLB(rigRef.current, config, binary);
        const kb = (sizeBytes / 1024).toFixed(1);
        setExportStatusMessage(`Exportado: ${filename} (${kb} KB)`);
        setTimeout(() => setExportStatusMessage(null), 5000);
      } catch (err) {
        console.error(err);
        setExportStatusMessage('Error al exportar el archivo GLB.');
      }
    },
    [config]
  );

  const threatScore = calculateThreatScore(config.stats);

  return (
    <div className="relative w-screen h-screen overflow-hidden bg-[#0b0d13] select-none">
      {/* 1. Strict 3-Zone Top Navigation Bar Contract */}
      <header className="fixed top-0 left-0 right-0 z-30 flex items-center justify-between gap-4 sm:gap-8 px-4 sm:px-6 py-3 bg-[#0b0d13]/95 border-b border-slate-800">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            handleCameraPresetChange('hero_three_quarter');
          }}
          className="font-display text-sm sm:text-base font-bold tracking-wider text-slate-100 whitespace-nowrap shrink-0"
        >
          AetherForge 3D
        </a>

        {/* Zone 2: 4 concise single-line navigation links (always visible) */}
        <nav className="flex items-center gap-3 sm:gap-6 text-xs font-semibold text-slate-300 overflow-x-auto">
          <button
            onClick={() => {
              setActiveTab('catalog');
              setIsDockCollapsed(false);
            }}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'catalog' ? 'text-amber-400 underline underline-offset-4' : ''
            }`}
          >
            Personajes (40)
          </button>
          <button
            onClick={() => {
              setActiveTab('anatomy');
              setIsDockCollapsed(false);
            }}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'anatomy' ? 'text-amber-400 underline underline-offset-4' : ''
            }`}
          >
            Morfología
          </button>
          <button
            onClick={() => {
              setActiveTab('equipment');
              setIsDockCollapsed(false);
            }}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'equipment' ? 'text-amber-400 underline underline-offset-4' : ''
            }`}
          >
            Materiales PBR
          </button>
          <button
            onClick={() => {
              setActiveTab('scene');
              setIsDockCollapsed(false);
            }}
            className={`hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              activeTab === 'scene' ? 'text-amber-400 underline underline-offset-4' : ''
            }`}
          >
            Cinemática
          </button>
        </nav>

        {/* Zone 3: 1 Primary Action */}
        <div className="flex items-center gap-3 shrink-0">
          {window.parent !== window && (
            <button
              onClick={() => window.parent.postMessage({ tipo: 'enemigo-disenado', config }, window.location.origin)}
              className="px-3 sm:px-4 py-2 text-xs font-semibold text-slate-950 bg-emerald-400 hover:bg-emerald-300 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              title="Enviar este enemigo al mundo que estás creando"
            >
              Usar en el mundo
            </button>
          )}
          <button
            onClick={() => setIsExportModalOpen(true)}
            className="px-3 sm:px-4 py-2 text-xs font-semibold text-slate-950 bg-amber-500 hover:bg-amber-400 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            Exportar (.GLB / Three.js)
          </button>
        </div>
      </header>

      {/* 2. Minimal Top-Left Unboxed Spec Readout */}
      <div className="fixed top-15 left-4 sm:left-6 z-20 space-y-1">
        <button
          onClick={() => {
            setActiveTab('catalog');
            setIsDockCollapsed(false);
          }}
          className="font-display text-base sm:text-xl font-bold text-slate-100 hover:text-amber-400 tracking-wide transition-colors cursor-pointer text-left"
          title="Haz clic para cambiar de personaje"
        >
          {config.name}
        </button>
        <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300 font-mono-tabular pointer-events-none">
          <span>{config.title}</span>
          <span aria-hidden="true">·</span>
          <span>{meshMetrics.partCount} piezas 3D</span>
          <span aria-hidden="true">·</span>
          <span>{meshMetrics.totalTriangles.toLocaleString()} tris</span>
          <span aria-hidden="true">·</span>
          <span>Amenaza {threatScore}</span>
          {exportStatusMessage && (
            <>
              <span aria-hidden="true">·</span>
              <span className="text-amber-400">{exportStatusMessage}</span>
            </>
          )}
        </div>
      </div>

      {/* 3. Isolated Full-Bleed WebGL Canvas Container */}
      <div
        ref={canvasMountRef}
        className="w-full h-full cursor-grab active:cursor-grabbing"
        aria-label="Visor interactivo 3D de enemigo RPG"
      />

      {/* WebGL Context Lost Resilient Fallback */}
      {webglLost && (
        <div className="fixed inset-0 z-40 flex items-center justify-center bg-[#0b0d13]/95 p-6 text-center">
          <div className="max-w-md space-y-2">
            <h2 className="font-display text-lg font-bold text-amber-400">
              Restaurando Contexto Gráfico WebGL
            </h2>
            <p className="text-xs text-slate-300">
              El navegador pausó temporalmente el lienzo 3D. Tus cambios morfológicos y atributos RPG están intactos.
            </p>
          </div>
        </div>
      )}

      {/* 4. Unified Collapsible Bottom Control Dock */}
      <BottomControlDock
        config={config}
        onChangeConfig={setConfig}
        activeTab={activeTab}
        onTabChange={setActiveTab}
        isCollapsed={isDockCollapsed}
        onToggleCollapse={() => setIsDockCollapsed((prev) => !prev)}
        explodedAmount={explodedAmount}
        onExplodedChange={setExplodedAmount}
        animState={animState}
        onAnimStateChange={setAnimState}
        animSpeed={animSpeed}
        onAnimSpeedChange={setAnimSpeed}
        cameraPreset={cameraPreset}
        onCameraPresetChange={handleCameraPresetChange}
        lightingPreset={lightingPreset}
        onLightingPresetChange={setLightingPreset}
        showHitboxes={showHitboxes}
        onToggleHitboxes={() => setShowHitboxes((prev) => !prev)}
        wireframe={wireframe}
        onToggleWireframe={() => setWireframe((prev) => !prev)}
        autoRotate={autoRotate}
        onToggleAutoRotate={() => setAutoRotate((prev) => !prev)}
        onRandomize={handleRandomize}
        onQuickExportGLB={() => handleExportGLB(true)}
        onOpenExportModal={() => setIsExportModalOpen(true)}
        inspectedPart={inspectedPart}
      />

      {/* 5. Export & Integration Pipeline Modal (.GLB + Three.js TS + JSON) */}
      <ExportModal
        isOpen={isExportModalOpen}
        onClose={() => setIsExportModalOpen(false)}
        config={config}
        metrics={meshMetrics}
        onExportGLB={handleExportGLB}
        onImportConfig={(imported) => setConfig(imported)}
        exportStatusMessage={exportStatusMessage}
      />
    </div>
  );
}
