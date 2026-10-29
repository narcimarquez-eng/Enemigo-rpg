import * as THREE from 'three';
import { GLTFExporter } from 'three/examples/jsm/exporters/GLTFExporter.js';
import { AuraParticleEffectId, EnemyConfig } from '../types/enemy';
import { BuiltEnemyRig, resetRigToBindPose } from './enemyBuilder';

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = '';
  for (let i = 0; i < bytes.length; i += chunkSize) {
    const chunk = bytes.subarray(i, i + chunkSize);
    binary += String.fromCharCode.apply(null, chunk as unknown as number[]);
  }
  return btoa(binary);
}

/**
 * Serializes the unified THREE.SkinnedMesh + THREE.Bone skeleton + 5 embedded THREE.AnimationClip
 * tracks ('Idle', 'Walk', 'Attack', 'Hit', 'Death') + Weapon_Slash_Trail_Arc + Elemental_Breath_Cone
 * + animated Orbiting Aura Shards & Procedural Aura Particle Motes into a .GLB ArrayBuffer or .GLTF JSON.
 */
export async function serializeEnemyToGLBData(
  rig: BuiltEnemyRig,
  auraParticleMode: AuraParticleEffectId = 'arcane_sparks',
  binary = true
): Promise<ArrayBuffer | object> {
  const exporter = new GLTFExporter();
  const targetGroup = rig.exportableGroup;

  // 1. Save current animated bone transforms & reset skeleton to clean bind pose
  const savedBoneState = rig.skeleton.bones.map((b) => ({
    bone: b,
    pos: b.position.clone(),
    quat: b.quaternion.clone(),
    scale: b.scale.clone(),
  }));
  resetRigToBindPose(rig);

  // 2. Temporarily detach AxesHelpers and hidden exploded-view source meshes so only unified SkinnedMesh + Skeleton export
  const detachedNodes: Array<{ parent: THREE.Object3D; child: THREE.Object3D }> = [];
  const toDetach: THREE.Object3D[] = [];
  targetGroup.traverse((child) => {
    if (
      (child as THREE.LineSegments).isLineSegments ||
      child.type === 'AxesHelper' ||
      child.userData?.isExplodedSourceMesh
    ) {
      toDetach.push(child);
    }
  });
  for (const node of toDetach) {
    if (node.parent) {
      detachedNodes.push({ parent: node.parent, child: node });
      node.parent.remove(node);
    }
  }

  // 3. Temporarily expand InstancedMesh into individual Meshes for universal GLTF 2.0 compatibility
  const instancedSwaps: Array<{
    parent: THREE.Object3D;
    inst: THREE.InstancedMesh;
    holder: THREE.Group;
  }> = [];
  const instancedNodes: THREE.InstancedMesh[] = [];
  targetGroup.traverse((child) => {
    if ((child as THREE.InstancedMesh).isInstancedMesh) {
      instancedNodes.push(child as THREE.InstancedMesh);
    }
  });

  const mat4 = new THREE.Matrix4();
  for (const inst of instancedNodes) {
    if (!inst.parent) continue;
    const holder = new THREE.Group();
    holder.name = inst.name || 'Aura_Shards_Group';
    for (let i = 0; i < inst.count; i++) {
      inst.getMatrixAt(i, mat4);
      const shardMesh = new THREE.Mesh(inst.geometry, inst.material);
      shardMesh.name = `Runic_Shard_${i}`;
      shardMesh.applyMatrix4(mat4);
      holder.add(shardMesh);
    }
    instancedSwaps.push({ parent: inst.parent, inst, holder });
    inst.parent.add(holder);
    inst.parent.remove(inst);
  }

  // 4. Create 16 3D Aura Particle Motes inside targetGroup so standard .GLB viewers also display animated Aura Particles!
  const glbParticleHolder = new THREE.Group();
  glbParticleHolder.name = 'Aura_Particles_GLB_Group';
  const isVolcanic = auraParticleMode === 'volcanic_smoke';
  const glbMoteGeo = isVolcanic
    ? new THREE.IcosahedronGeometry(0.055, 1)
    : new THREE.OctahedronGeometry(0.042, 0);
  const glbMoteMat = new THREE.MeshStandardMaterial({
    name: isVolcanic ? 'Mat_GLB_VolcanicSmoke' : 'Mat_GLB_ArcaneSparks',
    color: isVolcanic
      ? new THREE.Color('#3b231e')
      : rig.materials.emissiveRuneMat.color.clone().lerp(new THREE.Color('#ffffff'), 0.35),
    emissive: isVolcanic
      ? new THREE.Color('#ff5511')
      : rig.materials.emissiveRuneMat.color.clone(),
    emissiveIntensity: isVolcanic ? 1.2 : 2.2,
    roughness: isVolcanic ? 0.85 : 0.15,
    transparent: true,
    opacity: auraParticleMode === 'none' ? 0.0 : 0.88,
  });

  const GLB_MOTE_COUNT = auraParticleMode === 'none' ? 0 : 16;
  for (let mIdx = 0; mIdx < GLB_MOTE_COUNT; mIdx++) {
    const moteMesh = new THREE.Mesh(glbMoteGeo, glbMoteMat);
    moteMesh.name = `Aura_Particle_Mote_${mIdx}`;
    glbParticleHolder.add(moteMesh);
  }
  if (GLB_MOTE_COUNT > 0) {
    targetGroup.add(glbParticleHolder);
  }

  // 5. Clonerig animations and append keyframe tracks for Runic_Shard_* and Aura_Particle_Mote_* so they animate in .GLB
  const exportAnimations = rig.animations.map((baseClip) => {
    const extraTracks: THREE.KeyframeTrack[] = [];
    const steps = 16;
    const dur = baseClip.duration;
    const times = new Float32Array(steps + 1);
    for (let s = 0; s <= steps; s++) {
      times[s] = Number(((s / steps) * dur).toFixed(4));
    }

    // Orbiting Runic Shards keyframe tracks
    for (let i = 0; i < rig.shardData.length; i++) {
      const sd = rig.shardData[i];
      const posBuf = new Float32Array((steps + 1) * 3);
      for (let s = 0; s <= steps; s++) {
        const frac = s / steps;
        const ang = sd.angle + frac * Math.PI * 2;
        const y = sd.yOffset + Math.sin(frac * Math.PI * 2 + i) * 0.12;
        posBuf[s * 3] = Math.cos(ang) * sd.radius;
        posBuf[s * 3 + 1] = y;
        posBuf[s * 3 + 2] = Math.sin(ang) * sd.radius;
      }
      extraTracks.push(new THREE.VectorKeyframeTrack(`Runic_Shard_${i}.position`, times, posBuf));
    }

    // Rising & spiraling Aura Particle Motes keyframe tracks from aura anchors
    for (let mIdx = 0; mIdx < GLB_MOTE_COUNT; mIdx++) {
      const posBuf = new Float32Array((steps + 1) * 3);
      const scaleBuf = new Float32Array((steps + 1) * 3);
      const baseAngle = (mIdx / GLB_MOTE_COUNT) * Math.PI * 2;
      const baseRadius = 0.45 + (mIdx % 4) * 0.22;
      const baseY = 0.35 + (mIdx % 5) * 0.32;

      for (let s = 0; s <= steps; s++) {
        const u = ((s / steps) + mIdx / GLB_MOTE_COUNT) % 1.0;
        const spiral = baseAngle + u * (isVolcanic ? 1.4 : 3.6);
        const r = baseRadius + (isVolcanic ? u * 0.24 : Math.sin(u * Math.PI) * 0.18);
        posBuf[s * 3] = Math.cos(spiral) * r;
        posBuf[s * 3 + 1] = baseY + u * (isVolcanic ? 0.95 : 0.72);
        posBuf[s * 3 + 2] = Math.sin(spiral) * r;

        const env = Math.sin(u * Math.PI) * (isVolcanic ? 1.35 : 1.1);
        const sc = Math.max(0.01, env);
        scaleBuf[s * 3] = sc;
        scaleBuf[s * 3 + 1] = sc;
        scaleBuf[s * 3 + 2] = sc;
      }
      extraTracks.push(
        new THREE.VectorKeyframeTrack(`Aura_Particle_Mote_${mIdx}.position`, times, posBuf)
      );
      extraTracks.push(
        new THREE.VectorKeyframeTrack(`Aura_Particle_Mote_${mIdx}.scale`, times, scaleBuf)
      );
    }

    return new THREE.AnimationClip(baseClip.name, baseClip.duration, [
      ...baseClip.tracks,
      ...extraTracks,
    ]);
  });

  targetGroup.updateMatrixWorld(true);

  const restoreSceneState = () => {
    if (GLB_MOTE_COUNT > 0) {
      targetGroup.remove(glbParticleHolder);
    }
    glbMoteGeo.dispose();
    glbMoteMat.dispose();
    for (const swap of instancedSwaps) {
      swap.parent.remove(swap.holder);
      swap.parent.add(swap.inst);
    }
    for (const item of detachedNodes) {
      item.parent.add(item.child);
    }
    for (const st of savedBoneState) {
      st.bone.position.copy(st.pos);
      st.bone.quaternion.copy(st.quat);
      st.bone.scale.copy(st.scale);
    }
    targetGroup.updateMatrixWorld(true);
  };

  return new Promise((resolve, reject) => {
    exporter.parse(
      targetGroup,
      (result) => {
        restoreSceneState();
        resolve(result);
      },
      (error) => {
        restoreSceneState();
        reject(error);
      },
      {
        binary,
        trs: true,
        onlyVisible: true,
        animations: exportAnimations,
      }
    );
  });
}

/**
 * Exports the unified THREE.SkinnedMesh + THREE.Bone skeleton + 5 embedded THREE.AnimationClip
 * tracks ('Idle', 'Walk', 'Attack', 'Hit', 'Death') + Weapon_Slash_Trail_Arc + Aura Particles
 * into a standard binary .GLB or .GLTF 2.0 file.
 */
export async function exportEnemyToGLB(
  rig: BuiltEnemyRig,
  config: EnemyConfig,
  binary = true,
  auraParticleMode: AuraParticleEffectId = 'arcane_sparks'
): Promise<{ sizeBytes: number; filename: string }> {
  const result = await serializeEnemyToGLBData(rig, auraParticleMode, binary);
  const ext = binary ? 'glb' : 'gltf';
  const filename = `${config.id}.${ext}`;
  if (result instanceof ArrayBuffer) {
    const blob = new Blob([result], { type: 'model/gltf-binary' });
    triggerDownload(blob, filename);
    return { sizeBytes: result.byteLength, filename };
  } else {
    const jsonStr = JSON.stringify(result, null, 2);
    const blob = new Blob([jsonStr], { type: 'model/gltf+json' });
    triggerDownload(blob, filename);
    return { sizeBytes: blob.size, filename };
  }
}

/**
 * Exports a 100% self-contained, single-file .HTML 3D Interactive Viewer & Embeddable Widget
 * containing the embedded Base64 .GLB (SkinnedMesh + Skeleton + 5 Animations + Weapon Slash Trail + PBR Textures)
 * plus the real-time Procedural Aura Particle Emitter (Volcanic Smoke & Arcane Sparks).
 */
export async function exportEnemyToHTML(
  rig: BuiltEnemyRig,
  config: EnemyConfig,
  initialParticleMode: AuraParticleEffectId = 'arcane_sparks'
): Promise<{ sizeBytes: number; filename: string }> {
  const glbBuffer = (await serializeEnemyToGLBData(rig, initialParticleMode, true)) as ArrayBuffer;
  const base64Glb = arrayBufferToBase64(glbBuffer);
  const filename = `${config.id}_viewer.html`;

  const htmlContent = `<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>${config.name} — Visor 3D Autónomo (.GLB + Partículas + Estela)</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      background: #0b0d13;
      color: #f1f5f9;
      font-family: system-ui, -apple-system, sans-serif;
      overflow: hidden;
      width: 100vw;
      height: 100vh;
      user-select: none;
    }
    #canvas-container { width: 100%; height: 100%; cursor: grab; }
    #canvas-container:active { cursor: grabbing; }
    .hud-top {
      position: fixed;
      top: 16px;
      left: 20px;
      z-index: 10;
      pointer-events: none;
    }
    .hud-top h1 {
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: 0.04em;
      color: #f8fafc;
    }
    .hud-top p {
      font-size: 0.78rem;
      color: #94a3b8;
      margin-top: 4px;
      font-family: ui-monospace, monospace;
    }
    .hud-dock {
      position: fixed;
      bottom: 16px;
      left: 50%;
      transform: translateX(-50%);
      z-index: 10;
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
      justify-content: center;
      background: rgba(15, 20, 32, 0.92);
      border: 1px solid #334155;
      padding: 10px 16px;
      border-radius: 12px;
      box-shadow: 0 20px 40px rgba(0,0,0,0.6);
      max-width: 95vw;
    }
    .btn-group {
      display: flex;
      align-items: center;
      gap: 6px;
      flex-wrap: wrap;
    }
    .group-label {
      font-size: 0.72rem;
      font-weight: 700;
      color: #cbd5e1;
      margin-right: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    button {
      background: #0d121d;
      color: #e2e8f0;
      border: 1px solid #334155;
      border-radius: 6px;
      padding: 6px 11px;
      font-size: 0.75rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    button:hover { border-color: #64748b; }
    button.active-anim {
      background: #f59e0b;
      color: #090d16;
      border-color: #fbbf24;
    }
    button.active-fx {
      background: #38bdf8;
      color: #090d16;
      border-color: #7dd3fc;
    }
    .divider {
      width: 1px;
      height: 24px;
      background: #334155;
      margin: 0 4px;
    }
  </style>
  <script type="importmap">
    {
      "imports": {
        "three": "https://unpkg.com/three@0.170.0/build/three.module.js",
        "three/addons/": "https://unpkg.com/three@0.170.0/examples/jsm/"
      }
    }
  </script>
</head>
<body>
  <div class="hud-top">
    <h1>${config.name}</h1>
    <p>${config.title} · Nv.${config.stats.level} ${config.stats.tier} · Elemento: ${config.stats.element} · HP ${config.stats.hp} · ATQ ${config.stats.attack}</p>
  </div>

  <div id="canvas-container"></div>

  <div class="hud-dock">
    <div class="btn-group" id="anim-buttons">
      <span class="group-label">Animación (.GLB):</span>
      <button data-clip="Idle" class="active-anim">Reposo</button>
      <button data-clip="Walk">Marcha Realista</button>
      <button data-clip="Attack">Ataque / Fuego</button>
      <button data-clip="JumpBite">Salto Mordida</button>
      <button data-clip="Hit">Impacto</button>
      <button data-clip="Death">Derrota</button>
    </div>
    <div class="divider"></div>
    <div class="btn-group" id="fx-buttons">
      <span class="group-label">Partículas Aura:</span>
      <button data-fx="none" class="${initialParticleMode === 'none' ? 'active-fx' : ''}">Desactivado</button>
      <button data-fx="volcanic_smoke" class="${initialParticleMode === 'volcanic_smoke' ? 'active-fx' : ''}">Humo Volcánico</button>
      <button data-fx="arcane_sparks" class="${initialParticleMode === 'arcane_sparks' ? 'active-fx' : ''}">Chispas Arcanas</button>
    </div>
  </div>

  <script type="module">
    import * as THREE from 'three';
    import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
    import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
    import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

    const BASE64_GLB = "${base64Glb}";
    const EMISSIVE_HEX = "${config.materials.emissiveColor}";
    let particleMode = "${initialParticleMode}";

    const container = document.getElementById('canvas-container');
    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0b0d13);
    scene.fog = new THREE.FogExp2(0x0b0d13, 0.04);

    const pmrem = new THREE.PMREMGenerator(renderer);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    pmrem.dispose();

    const camera = new THREE.PerspectiveCamera(42, window.innerWidth / window.innerHeight, 0.1, 80);
    camera.position.set(2.85, 2.25, 3.65);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 1.35, 0);
    controls.enableDamping = true;
    controls.update();

    const keyLight = new THREE.DirectionalLight(0xffdca8, 2.6);
    keyLight.position.set(4.2, 6.8, 4.5);
    keyLight.castShadow = true;
    scene.add(keyLight);

    const hemiLight = new THREE.HemisphereLight(0x38bdf8, 0x1e130c, 0.9);
    scene.add(hemiLight);

    const rimLight = new THREE.SpotLight(0xf59e0b, 2.4, 18, Math.PI / 3, 0.6, 1.5);
    rimLight.position.set(-3.5, 4.2, -4.0);
    scene.add(rimLight);

    // Pedestal
    const dais = new THREE.Mesh(
      new THREE.CylinderGeometry(2.35, 2.65, 0.22, 24),
      new THREE.MeshStandardMaterial({ color: 0x121620, metalness: 0.55, roughness: 0.38 })
    );
    dais.position.y = -0.11;
    dais.receiveShadow = true;
    scene.add(dais);

    // Procedural Particle Sprite Textures
    function makeSpriteTex(isSpark) {
      const c = document.createElement('canvas');
      c.width = 64; c.height = 64;
      const ctx = c.getContext('2d');
      const g = ctx.createRadialGradient(32, 32, 1, 32, 32, 30);
      if (isSpark) {
        g.addColorStop(0, 'rgba(255,255,255,1)');
        g.addColorStop(0.3, 'rgba(255,255,255,0.8)');
        g.addColorStop(1, 'rgba(255,255,255,0)');
      } else {
        g.addColorStop(0, 'rgba(255,255,255,0.9)');
        g.addColorStop(0.5, 'rgba(200,190,185,0.45)');
        g.addColorStop(1, 'rgba(0,0,0,0)');
      }
      ctx.fillStyle = g;
      ctx.fillRect(0, 0, 64, 64);
      const tex = new THREE.CanvasTexture(c);
      tex.colorSpace = THREE.SRGBColorSpace;
      return tex;
    }

    const PARTICLE_COUNT = 140;
    const pGeo = new THREE.BufferGeometry();
    const pPos = new Float32Array(PARTICLE_COUNT * 3);
    const pCol = new Float32Array(PARTICLE_COUNT * 3);
    pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
    pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));

    const smokeMat = new THREE.PointsMaterial({
      map: makeSpriteTex(false), size: 0.36, transparent: true, opacity: 0.82, depthWrite: false, vertexColors: true
    });
    const sparkMat = new THREE.PointsMaterial({
      map: makeSpriteTex(true), size: 0.24, transparent: true, opacity: 0.95, depthWrite: false, vertexColors: true, blending: THREE.AdditiveBlending
    });
    const particlePoints = new THREE.Points(pGeo, sparkMat);
    particlePoints.frustumCulled = false;
    scene.add(particlePoints);

    const pool = Array.from({ length: PARTICLE_COUNT }, (_, i) => ({
      age: (i / PARTICLE_COUNT) * 1.2,
      lifetime: 0.8 + (i % 9) * 0.12,
      anchorIdx: i,
      vx: 0, vy: 0.55, vz: 0,
      seed: i * 1.618
    }));

    // Decode embedded Base64 .GLB
    const binStr = atob(BASE64_GLB);
    const len = binStr.length;
    const bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) bytes[i] = binStr.charCodeAt(i);

    let mixer = null;
    const actions = {};
    let activeAction = null;
    const anchorNodes = [];

    const loader = new GLTFLoader();
    loader.parse(bytes.buffer, '', (gltf) => {
      const model = gltf.scene;
      model.traverse((obj) => {
        if (obj.isMesh) {
          obj.castShadow = true;
          obj.receiveShadow = true;
        }
        if (obj.name.startsWith('socket_') || obj.name.startsWith('Runic_Shard_')) {
          anchorNodes.push(obj);
        }
      });
      scene.add(model);

      mixer = new THREE.AnimationMixer(model);
      gltf.animations.forEach((clip) => {
        actions[clip.name] = mixer.clipAction(clip);
      });
      if (actions['Idle']) {
        activeAction = actions['Idle'];
        activeAction.play();
      }
    });

    document.querySelectorAll('#anim-buttons button').forEach((btn) => {
      btn.addEventListener('click', () => {
        const clipName = btn.getAttribute('data-clip');
        if (!actions[clipName]) return;
        document.querySelectorAll('#anim-buttons button').forEach((b) => b.classList.remove('active-anim'));
        btn.classList.add('active-anim');
        const nextAction = actions[clipName];
        if (activeAction && activeAction !== nextAction) {
          activeAction.fadeOut(0.2);
        }
        nextAction.reset().fadeIn(0.2).play();
        activeAction = nextAction;
      });
    });

    document.querySelectorAll('#fx-buttons button').forEach((btn) => {
      btn.addEventListener('click', () => {
        particleMode = btn.getAttribute('data-fx');
        document.querySelectorAll('#fx-buttons button').forEach((b) => b.classList.remove('active-fx'));
        btn.classList.add('active-fx');
      });
    });

    const clock = new THREE.Clock();
    const tmpVec = new THREE.Vector3();
    const baseCol = new THREE.Color(EMISSIVE_HEX);
    const outCol = new THREE.Color();

    function animate() {
      requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.1);
      const t = clock.getElapsedTime();
      if (mixer) mixer.update(dt);

      if (particleMode === 'none' || anchorNodes.length === 0) {
        particlePoints.visible = false;
      } else {
        particlePoints.visible = true;
        const isVolc = particleMode === 'volcanic_smoke';
        particlePoints.material = isVolc ? smokeMat : sparkMat;
        const posAttr = pGeo.getAttribute('position');
        const colAttr = pGeo.getAttribute('color');

        for (let i = 0; i < PARTICLE_COUNT; i++) {
          const p = pool[i];
          p.age += dt;
          if (p.age >= p.lifetime) {
            p.age = 0;
            const node = anchorNodes[p.anchorIdx % anchorNodes.length];
            p.anchorIdx++;
            node.getWorldPosition(tmpVec);
            const ang = t * 3.5 + p.seed * 6.28;
            posAttr.setXYZ(i, tmpVec.x + Math.cos(ang) * 0.07, tmpVec.y, tmpVec.z + Math.sin(ang) * 0.07);
            p.vx = Math.cos(ang) * (isVolc ? 0.16 : 0.38);
            p.vy = 0.5 + (i % 7) * 0.06;
            p.vz = Math.sin(ang) * (isVolc ? 0.16 : 0.38);
          }
          const ratio = p.age / p.lifetime;
          let x = posAttr.getX(i) + (p.vx + Math.sin(t * 3 + p.seed) * 0.22) * dt;
          let y = posAttr.getY(i) + p.vy * dt;
          let z = posAttr.getZ(i) + (p.vz + Math.cos(t * 3 + p.seed) * 0.22) * dt;
          posAttr.setXYZ(i, x, y, z);

          if (isVolc) {
            outCol.set(ratio < 0.28 ? '#ff5511' : '#261d1e').multiplyScalar(Math.max(0, 1 - ratio * 0.85));
          } else {
            outCol.copy(baseCol).lerp(new THREE.Color('#ffffff'), ratio < 0.3 ? 0.5 : 0.1).multiplyScalar(Math.max(0, 1 - ratio));
          }
          colAttr.setXYZ(i, outCol.r, outCol.g, outCol.b);
        }
        posAttr.needsUpdate = true;
        colAttr.needsUpdate = true;
      }

      controls.update();
      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      camera.aspect = window.innerWidth / window.innerHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(window.innerWidth, window.innerHeight);
    });
  </script>
</body>
</html>`;

  const blob = new Blob([htmlContent], { type: 'text/html;charset=utf-8' });
  triggerDownload(blob, filename);
  return { sizeBytes: blob.size, filename };
}

export function downloadTextFile(content: string, filename: string, mimeType = 'text/plain') {
  const blob = new Blob([content], { type: mimeType });
  triggerDownload(blob, filename);
}

/**
 * Generates a standalone, zero-dependency Three.js / TypeScript module that builds
 * the exact composite enemy (Dragon, Beast, Mythological, Magical, or Biped), sockets,
 * Weapon Slash Trail Arc, Procedural Aura Particles, RPG stats, and procedural animation controller.
 */
export function generateThreeJsModuleCode(config: EnemyConfig): string {
  const { morphology: m, materials: mat, stats, bodyPlan } = config;
  const isQuadruped = bodyPlan === 'quadruped_beast' || bodyPlan === 'dragon_wyvern';
  return `/**
 * Auto-generated by AetherForge 3D — RPG Enemy Studio
 * Enemy: ${config.name} (${config.title})
 * Race Category: ${config.raceCategory} | Body Plan: ${config.bodyPlan}
 * Archetype: ${config.archetype} | Tier: ${stats.tier} | Level: ${stats.level}
 *
 * Includes:
 * - Hierarchical THREE.Bone Skeleton & Sockets (socket_head, socket_weapon_r, socket_shield_l, socket_core)
 * - Weapon Slash Trail Arc (Weapon_Slash_Trail_Arc) activated during 'attack'
 * - Procedural Aura Particle Emitter ('none' | 'volcanic_smoke' | 'arcane_sparks')
 */
import * as THREE from 'three';

export type EnemyAnimState = 'idle' | 'walk' | 'attack' | 'hit' | 'death';
export type EnemyAuraParticleMode = 'none' | 'volcanic_smoke' | 'arcane_sparks';

export const ENEMY_RPG_DATA = ${JSON.stringify(
    {
      id: config.id,
      name: config.name,
      title: config.title,
      raceCategory: config.raceCategory,
      bodyPlan: config.bodyPlan,
      archetype: config.archetype,
      stats: config.stats,
    },
    null,
    2
  )} as const;

export interface EnemyInstanceRig {
  group: THREE.Group;
  skeleton: THREE.Skeleton;
  sockets: {
    head: THREE.Object3D;
    weaponR: THREE.Object3D;
    shieldL: THREE.Object3D;
    core: THREE.Object3D;
  };
  parts: {
    rootBone: THREE.Bone;
    pelvis: THREE.Bone;
    torso: THREE.Bone;
    neck: THREE.Bone;
    head: THREE.Bone;
    tail: THREE.Bone;
    tailTip: THREE.Bone;
    leftWing: THREE.Bone;
    rightWing: THREE.Bone;
    leftArm: THREE.Bone;
    leftElbow: THREE.Bone;
    rightArm: THREE.Bone;
    rightElbow: THREE.Bone;
    leftLeg: THREE.Bone;
    leftKnee: THREE.Bone;
    rightLeg: THREE.Bone;
    rightKnee: THREE.Bone;
    coreMesh: THREE.Mesh;
    slashTrailMesh: THREE.Mesh;
    auraParticles: THREE.Points;
  };
  update: (
    state: EnemyAnimState,
    elapsed: number,
    delta: number,
    particleMode?: EnemyAuraParticleMode
  ) => void;
  dispose: () => void;
}

export function createEnemy_${config.id.replace(/[^a-zA-Z0-9_]/g, '_')}(): EnemyInstanceRig {
  const group = new THREE.Group();
  group.name = '${config.id}';
  group.userData = { ...ENEMY_RPG_DATA };

  const geometries: THREE.BufferGeometry[] = [];
  const materials: THREE.Material[] = [];
  const trackGeo = <T extends THREE.BufferGeometry>(g: T): T => { geometries.push(g); return g; };
  const trackMat = <T extends THREE.Material>(matItem: T): T => { materials.push(matItem); return matItem; };
  const makeBone = (name: string): THREE.Bone => { const b = new THREE.Bone(); b.name = name; return b; };

  // PBR Studio Materials with Organic Sheen
  const primaryMat = trackMat(new THREE.MeshPhysicalMaterial({
    color: '${mat.primaryColor}',
    metalness: ${mat.metalness},
    roughness: ${mat.roughness},
    anisotropy: ${mat.anisotropy},
    sheen: 0.55,
    sheenRoughness: 0.5,
    clearcoat: 0.25,
  }));

  const secondaryMat = trackMat(new THREE.MeshPhysicalMaterial({
    color: '${mat.secondaryColor}',
    metalness: ${Math.min(1, Number((mat.metalness + 0.12).toFixed(2)))},
    roughness: ${Math.max(0.08, Number(((mat.roughness + 0.32) % 0.85).toFixed(2)))},
  }));

  const underbodyMat = trackMat(new THREE.MeshStandardMaterial({
    color: '${mat.underbodyColor}',
    metalness: ${Math.max(0.05, Number((mat.metalness * 0.25).toFixed(2)))},
    roughness: ${Math.min(0.92, Number((mat.roughness + 0.45).toFixed(2)))},
  }));

  const emissiveMat = trackMat(new THREE.MeshStandardMaterial({
    color: '${mat.emissiveColor}',
    emissive: '${mat.emissiveColor}',
    emissiveIntensity: ${mat.emissiveIntensity},
    roughness: 0.15,
  }));

  const slashTrailMat = trackMat(new THREE.MeshStandardMaterial({
    color: '${mat.emissiveColor}',
    emissive: '${mat.emissiveColor}',
    emissiveIntensity: ${(mat.emissiveIntensity * 1.45).toFixed(2)},
    transparent: true,
    opacity: 0.88,
    side: THREE.DoubleSide,
    depthWrite: false,
  }));

  const legHeight = ${(0.95 * m.limbLength).toFixed(3)};
  const torsoHeight = ${(0.92 * m.heightScale).toFixed(3)};
  const bulk = ${m.torsoBulk.toFixed(3)};
  const shoulderW = ${m.shoulderWidth.toFixed(3)};

  // 1. Hierarchical THREE.Bone Skeleton (Grounded at y = 0)
  const rootBone = makeBone('Root_Bone');
  group.add(rootBone);

  const pelvis = makeBone('Pelvis_Chassis');
  pelvis.position.set(0, legHeight, ${isQuadruped ? -0.35 : 0});
  rootBone.add(pelvis);
  pelvis.add(new THREE.Mesh(trackGeo(new THREE.CylinderGeometry(0.34 * bulk, 0.28 * bulk, 0.44, 12)), primaryMat));

  const buildLeg = (side: number, tag: string) => {
    const leg = makeBone('Leg_' + tag);
    leg.position.set(side * 0.28 * bulk, legHeight, ${isQuadruped ? -0.45 : 0});
    rootBone.add(leg);

    const thigh = new THREE.Mesh(trackGeo(new THREE.CylinderGeometry(0.16 * bulk, 0.12 * bulk, legHeight * 0.5, 10)), primaryMat);
    thigh.position.y = -legHeight * 0.24;
    leg.add(thigh);

    const knee = makeBone('Knee_Hock_' + tag);
    knee.position.set(0, -legHeight * 0.48, 0);
    leg.add(knee);

    const shin = new THREE.Mesh(trackGeo(new THREE.CylinderGeometry(0.12 * bulk, 0.09 * bulk, legHeight * 0.5, 10)), primaryMat);
    shin.position.y = -legHeight * 0.24;
    knee.add(shin);

    const pawGeo = trackGeo(new THREE.SphereGeometry(0.14 * bulk, 8, 6));
    pawGeo.scale(0.9, 0.5, 1.45);
    const paw = new THREE.Mesh(pawGeo, secondaryMat);
    paw.position.set(0, -legHeight * 0.46, 0.1);
    knee.add(paw);

    return { leg, knee };
  };
  const { leg: leftLeg, knee: leftKnee } = buildLeg(1, 'L');
  const { leg: rightLeg, knee: rightKnee } = buildLeg(-1, 'R');

  // 2. Torso & Elemental Core
  const torso = makeBone('Torso_Main');
  torso.position.set(0, 0.2, ${isQuadruped ? 0.45 : 0});
  torso.rotation.x = ${isQuadruped ? 0.38 : 0};
  pelvis.add(torso);

  const cuirassPts = [
    new THREE.Vector2(0.28 * bulk, 0),
    new THREE.Vector2(0.4 * bulk, torsoHeight * 0.35),
    new THREE.Vector2(0.52 * bulk * shoulderW * 0.85, torsoHeight * 0.78),
    new THREE.Vector2(0.32 * bulk, torsoHeight),
  ];
  const cuirassGeo = trackGeo(new THREE.LatheGeometry(cuirassPts, 12));
  cuirassGeo.scale(1.15, 1.0, 0.82);
  torso.add(new THREE.Mesh(cuirassGeo, primaryMat));

  const socketCore = new THREE.Object3D();
  socketCore.name = 'socket_core';
  socketCore.position.set(0, torsoHeight * 0.54, 0.42 * bulk);
  torso.add(socketCore);

  const coreMesh = new THREE.Mesh(trackGeo(new THREE.OctahedronGeometry(0.15 * bulk, 1)), emissiveMat);
  coreMesh.position.copy(socketCore.position);
  torso.add(coreMesh);

  // 3. Articulated Neck, Organic Sculpted Skull, Snout & Horns
  const neck = makeBone('Neck_Assembly');
  neck.position.set(0, torsoHeight * 0.92, 0.06);
  neck.rotation.x = ${isQuadruped ? -0.35 : 0};
  torso.add(neck);

  const head = makeBone('Head_Helm_Assembly');
  head.position.set(0, ${(0.45 * m.neckLength).toFixed(2)}, 0.1);
  neck.add(head);

  const socketHead = new THREE.Object3D();
  socketHead.name = 'socket_head';
  socketHead.position.set(0, 0.2, ${(0.28 * m.snoutLength).toFixed(2)});
  head.add(socketHead);

  const skullGeo = trackGeo(new THREE.SphereGeometry(0.23, 12, 10));
  skullGeo.scale(0.95, 0.88, 1.35);
  head.add(new THREE.Mesh(skullGeo, primaryMat));

  for (const side of [-1, 1]) {
    const hornCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(side * 0.16, 0.26, 0.02),
      new THREE.Vector3(side * ${(0.34 + 0.42 * m.hornLength * 0.4).toFixed(2)}, ${(0.34 + 0.42 * m.hornLength * 0.35).toFixed(2)}, -0.12),
      new THREE.Vector3(side * 0.24, ${(0.58 + 0.42 * m.hornLength * 1.1).toFixed(2)}, ${(0.32 * m.hornCurve).toFixed(2)}),
    ]);
    head.add(new THREE.Mesh(trackGeo(new THREE.TubeGeometry(hornCurve, 12, 0.048, 6, false)), secondaryMat));
  }

  // 4. Articulated Wings & 2-Bone Tail
  const leftWing = makeBone('Wing_L');
  const rightWing = makeBone('Wing_R');
  torso.add(leftWing, rightWing);

  const tail = makeBone('Articulated_Tail');
  tail.position.set(0, 0.05, -0.45);
  pelvis.add(tail);
  const tailTip = makeBone('Tail_Mid_Tip');
  tailTip.position.set(0, -0.1, ${(-0.55 * m.tailLength).toFixed(2)});
  tail.add(tailTip);

  if (${m.tailLength > 0.08 ? 'true' : 'false'}) {
    const tailCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0, 0),
      new THREE.Vector3(0, -0.15, ${(-0.55 * m.tailLength).toFixed(2)}),
      new THREE.Vector3(0, 0.1, ${(-1.2 * m.tailLength).toFixed(2)}),
    ]);
    tail.add(new THREE.Mesh(trackGeo(new THREE.TubeGeometry(tailCurve, 14, 0.09 * bulk, 7, false)), primaryMat));
  }

  // 5. 2-Bone Forelimbs / Arms & Equipment Sockets
  const buildArm = (side: number, tag: string, socketName: string) => {
    const arm = makeBone('Arm_Or_Foreleg_' + tag);
    arm.position.set(side * 0.52 * bulk * shoulderW, torsoHeight * 0.72, 0);
    torso.add(arm);

    const upperArm = new THREE.Mesh(trackGeo(new THREE.CylinderGeometry(0.12, 0.1, ${(0.38 * m.limbLength).toFixed(2)}, 8)), underbodyMat);
    upperArm.position.set(0, -0.19, 0);
    arm.add(upperArm);

    const elbow = makeBone('Elbow_Joint_' + tag);
    elbow.position.set(0, -0.38, 0);
    arm.add(elbow);

    const forearm = new THREE.Mesh(trackGeo(new THREE.CylinderGeometry(0.11, 0.09, ${(0.38 * m.limbLength).toFixed(2)}, 8)), primaryMat);
    forearm.position.set(0, -0.19, 0.04);
    elbow.add(forearm);

    const socket = new THREE.Object3D();
    socket.name = socketName;
    socket.position.set(0, -0.38, 0.14);
    elbow.add(socket);
    return { arm, elbow, socket };
  };

  const { arm: leftArm, elbow: leftElbow, socket: socketShieldL } = buildArm(1, 'L', 'socket_shield_l');
  const { arm: rightArm, elbow: rightElbow, socket: socketWeaponR } = buildArm(-1, 'R', 'socket_weapon_r');

  // 6. Weapon Slash Trail Arc on socket_weapon_r
  const slashGeo = trackGeo(new THREE.RingGeometry(0.48, 0.96, 18, 1, -Math.PI * 0.4, Math.PI * 0.85));
  slashGeo.rotateY(Math.PI / 2);
  const slashTrailMesh = new THREE.Mesh(slashGeo, slashTrailMat);
  slashTrailMesh.name = 'Weapon_Slash_Trail_Arc';
  slashTrailMesh.position.set(0, 0.08, 0.28);
  slashTrailMesh.scale.setScalar(0.0001);
  socketWeaponR.add(slashTrailMesh);

  // 7. Procedural Aura Particle Emitter around sockets
  const PARTICLE_COUNT = 96;
  const pGeo = trackGeo(new THREE.BufferGeometry());
  const pPos = new Float32Array(PARTICLE_COUNT * 3);
  const pCol = new Float32Array(PARTICLE_COUNT * 3);
  pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
  pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
  const pMat = trackMat(new THREE.PointsMaterial({
    size: 0.22,
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    vertexColors: true,
    blending: THREE.AdditiveBlending,
  }));
  const auraParticles = new THREE.Points(pGeo, pMat);
  auraParticles.name = 'Procedural_Aura_Particles';
  group.add(auraParticles);

  const bones: THREE.Bone[] = [];
  rootBone.traverse((c) => { if ((c as THREE.Bone).isBone) bones.push(c as THREE.Bone); });
  const skeleton = new THREE.Skeleton(bones);

  return {
    group,
    skeleton,
    sockets: { head: socketHead, weaponR: socketWeaponR, shieldL: socketShieldL, core: socketCore },
    parts: {
      rootBone, pelvis, torso, neck, head, tail, tailTip, leftWing, rightWing,
      leftArm, leftElbow, rightArm, rightElbow, leftLeg, leftKnee, rightLeg, rightKnee,
      coreMesh, slashTrailMesh, auraParticles,
    },
    update(
      state: EnemyAnimState,
      elapsed: number,
      delta: number,
      particleMode: EnemyAuraParticleMode = 'arcane_sparks'
    ) {
      coreMesh.rotation.y = elapsed * 1.5;
      tail.rotation.y = Math.sin(elapsed * 2.5) * 0.26;
      tailTip.rotation.y = Math.sin(elapsed * 2.5 - 0.9) * 0.35;
      const wingFlap = Math.sin(elapsed * 3.2) * 0.4;
      leftWing.rotation.y = wingFlap;
      rightWing.rotation.y = -wingFlap;
      const damp = (cur: number, target: number) => THREE.MathUtils.damp(cur, target, 10, delta);
      if (state === 'idle') {
        slashTrailMesh.scale.setScalar(0.0001);
        const b = Math.sin(elapsed * 2.3);
        const s = 1.0 + b * 0.025;
        torso.scale.set(s, 1.0 + b * 0.015, s);
        head.rotation.y = damp(head.rotation.y, Math.sin(elapsed * 1.15) * 0.12);
        rightArm.rotation.x = damp(rightArm.rotation.x, -0.18 + b * 0.06);
        leftArm.rotation.x = damp(leftArm.rotation.x, -0.22 - b * 0.06);
        leftElbow.rotation.x = damp(leftElbow.rotation.x, -0.32);
        rightElbow.rotation.x = damp(rightElbow.rotation.x, -0.32);
        leftKnee.rotation.x = damp(leftKnee.rotation.x, 0.16);
        rightKnee.rotation.x = damp(rightKnee.rotation.x, 0.16);
      } else if (state === 'walk') {
        slashTrailMesh.scale.setScalar(0.0001);
        const s = Math.sin(elapsed * 5.8);
        leftLeg.rotation.x = damp(leftLeg.rotation.x, s * 0.52);
        rightLeg.rotation.x = damp(rightLeg.rotation.x, -s * 0.52);
        leftKnee.rotation.x = damp(leftKnee.rotation.x, Math.max(0.08, -s * 0.65 + 0.25));
        rightKnee.rotation.x = damp(rightKnee.rotation.x, Math.max(0.08, s * 0.65 + 0.25));
        rightArm.rotation.x = damp(rightArm.rotation.x, s * 0.48 - 0.15);
        leftArm.rotation.x = damp(leftArm.rotation.x, -s * 0.48 - 0.15);
      } else if (state === 'attack') {
        const cycle = (elapsed * 2.5) % (Math.PI * 2);
        const swing = Math.sin(cycle);
        const slashPower = Math.max(0, Math.sin(cycle - 0.25));
        rightArm.rotation.x = damp(rightArm.rotation.x, swing > 0 ? -1.4 + swing * 2.2 : -0.35);
        rightElbow.rotation.x = damp(rightElbow.rotation.x, swing > 0 ? -0.6 + swing * 0.5 : -0.4);
        if (slashPower > 0.05) {
          slashTrailMesh.scale.set(1.0, slashPower * 1.35, slashPower * 1.35);
          slashTrailMesh.rotation.x = -0.35 + slashPower * 1.15;
        } else {
          slashTrailMesh.scale.setScalar(0.0001);
        }
      } else {
        slashTrailMesh.scale.setScalar(0.0001);
      }

      auraParticles.visible = particleMode !== 'none';
    },
    dispose() {
      geometries.forEach((g) => g.dispose());
      materials.forEach((mItem) => mItem.dispose());
    },
  };
}
`;
}

/**
 * Generates a ready-to-run Blender 3.6 / 4.x Python (bpy) post-processing script
 * that enhances the exported .GLB inside Blender with:
 *  1. Weld + Corrective Smooth + Catmull-Clark Subdivision (SUBSURF) modifiers on SkinnedMeshes
 *  2. Principled BSDF Subsurface Scattering (SSS) + Geometry Pointiness Curvature Edge Wear + Procedural Bump
 *  3. Emissive Bloom setup for Weapon_Slash_Trail_Arc, Elemental_Breath_Cone & Aura_Particles_GLB_Group
 *  4. Automatic re-export option to a subdivided game-ready .GLB
 */
export function generateBlenderPythonScript(config: EnemyConfig): string {
  const mat = config.materials;
  const wear = (mat.grainWearIntensity ?? 1.0).toFixed(2);

  return `# ==============================================================================
# AetherForge 3D — Blender 3.6 / 4.x Post-Processing & Sculpt Enhancer (bpy)
# Criatura: ${config.name} (${config.id})
#
# CÓMO USAR EN BLENDER:
# 1. Importa "${config.id}.glb" en Blender (File -> Import -> glTF 2.0)
#    o coloca "${config.id}.glb" en la misma carpeta y configura GLB_INPUT_PATH.
# 2. Abre la pestaña "Scripting" en Blender, crea un texto nuevo, pega este
#    script y pulsa "Run Script" (Alt + P).
# 3. También puedes ejecutarlo por consola sin interfaz (Headless CLI):
#    blender -b -P enhance_${config.id}_blender.py
# ==============================================================================

import bpy
import os

# Ruta opcional si deseas que el script importe y re-exporte automáticamente:
GLB_INPUT_PATH = "${config.id}.glb"
GLB_OUTPUT_PATH = "${config.id}_blender_enhanced.glb"
AUTO_EXPORT_ENHANCED_GLB = False  # Cambia a True para exportar automáticamente

PRIMARY_HEX = "${mat.primaryColor}"
SECONDARY_HEX = "${mat.secondaryColor}"
EMISSIVE_HEX = "${mat.emissiveColor}"
ROUGHNESS_VAL = ${mat.roughness.toFixed(2)}
METALNESS_VAL = ${mat.metalness.toFixed(2)}
GRAIN_WEAR_STRENGTH = ${wear}
EMISSIVE_STRENGTH = ${(mat.emissiveIntensity * 4.5).toFixed(2)}


def hex_to_rgba(hex_str, alpha=1.0):
    h = hex_str.lstrip('#')
    if len(h) != 6:
        return (0.5, 0.5, 0.5, alpha)
    # Conversión aproximada sRGB -> Linear para nodos de Blender
    r = pow(int(h[0:2], 16) / 255.0, 2.2)
    g = pow(int(h[2:4], 16) / 255.0, 2.2)
    b = pow(int(h[4:6], 16) / 255.0, 2.2)
    return (r, g, b, alpha)


def ensure_glb_imported():
    has_skinned = any(obj.type == 'MESH' and obj.name.startswith('Skinned_Unified_') for obj in bpy.data.objects)
    if not has_skinned and os.path.exists(GLB_INPUT_PATH):
        print(f"[AetherForge] Importando {GLB_INPUT_PATH}...")
        bpy.ops.import_scene.gltf(filepath=GLB_INPUT_PATH)


def apply_sculpt_and_subsurf_modifiers():
    """
    Aplica Weld (costuras UV), Corrective Smooth (deformación articular orgánica sin pellizcos)
    y Subdivision Surface Catmull-Clark sobre las mallas SkinnedMesh manteniendo el Armature.
    """
    processed = 0
    for obj in bpy.data.objects:
        if obj.type != 'MESH':
            continue

        is_character_skin = obj.name.startswith('Skinned_Unified_')
        if not is_character_skin:
            continue

        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)

        # Sombreado suave (Shade Smooth)
        for poly in obj.data.polygons:
            poly.use_smooth = True

        # 1. Modificador WELD para fusionar vértices coincidentes en costuras UV (sin romper pesos)
        if "AF_Weld_Seams" not in obj.modifiers:
            weld = obj.modifiers.new(name="AF_Weld_Seams", type='WELD')
            weld.merge_threshold = 0.0015
            if hasattr(weld, "mode"):
                weld.mode = 'ALL'

        # 2. Modificador CORRECTIVE SMOOTH para articulaciones orgánicas al animar el Armature
        if "AF_Corrective_Sculpt" not in obj.modifiers:
            cs = obj.modifiers.new(name="AF_Corrective_Sculpt", type='CORRECTIVE_SMOOTH')
            cs.factor = 0.58
            cs.iterations = 8
            cs.smooth_type = 'LENGTH_WEIGHTED'
            cs.use_only_smooth = True

        # 3. Modificador SUBSURF (Catmull-Clark) para silueta de alta resolución tipo escultura
        if "AF_Catmull_Subsurf" not in obj.modifiers:
            sub = obj.modifiers.new(name="AF_Catmull_Subsurf", type='SUBSURF')
            sub.levels = 1
            sub.render_levels = 2
            sub.uv_smooth = 'PRESERVE_BOUNDARIES'

        obj.select_set(False)
        processed += 1

    print(f"[AetherForge] Modificadores Corrective Smooth + Subsurf aplicados a {processed} mallas.")


def enhance_pbr_materials_with_curvature_and_sss():
    """
    Enriquece los materiales importados del .GLB añadiendo:
    - Subsurface Scattering (SSS) epidérmico/quitinoso en Principled BSDF
    - Desgaste de bordes por curvatura (Geometry -> Pointiness + Ambient Occlusion)
    - Micro-relieve procedural 3D (Noise Texture -> Bump) controlado por GRAIN_WEAR_STRENGTH
    """
    sec_rgba = hex_to_rgba(SECONDARY_HEX)
    emi_rgba = hex_to_rgba(EMISSIVE_HEX)

    for mat in bpy.data.materials:
        if not mat.use_nodes or not mat.node_tree:
            continue

        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        bsdf = next((n for n in nodes if n.type == 'BSDF_PRINCIPLED'), None)
        if not bsdf:
            continue

        # Evitar duplicar nodos si el script se ejecuta varias veces
        if nodes.get("AF_Curvature_Geo"):
            continue

        # Configurar Subsurface Scattering (compatible con Blender 3.6 y Blender 4.0+)
        if "Subsurface Weight" in bsdf.inputs:
            bsdf.inputs["Subsurface Weight"].default_value = 0.18
        elif "Subsurface" in bsdf.inputs:
            bsdf.inputs["Subsurface"].default_value = 0.18

        if "Subsurface Scale" in bsdf.inputs:
            bsdf.inputs["Subsurface Scale"].default_value = 0.045

        # Nodo de Geometría (Pointiness / Curvatura para desgaste de aristas y escamas)
        geo_node = nodes.new(type='ShaderNodeNewGeometry')
        geo_node.name = "AF_Curvature_Geo"
        geo_node.location = (-850, 220)

        ramp = nodes.new(type='ShaderNodeValToRGB')
        ramp.name = "AF_EdgeWear_Ramp"
        ramp.location = (-620, 220)
        ramp.color_ramp.elements[0].position = 0.47
        ramp.color_ramp.elements[0].color = (0.0, 0.0, 0.0, 1.0)
        ramp.color_ramp.elements[1].position = 0.56
        ramp.color_ramp.elements[1].color = (1.0, 1.0, 1.0, 1.0)
        links.new(geo_node.outputs["Pointiness"], ramp.inputs["Fac"])

        # Nodo de Ruido 3D para micro-poros, cicatrices y grano orgánico
        noise = nodes.new(type='ShaderNodeTexNoise')
        noise.name = "AF_Organic_Grain_Noise"
        noise.location = (-850, -180)
        noise.inputs["Scale"].default_value = 42.0
        noise.inputs["Detail"].default_value = 8.0
        noise.inputs["Roughness"].default_value = 0.68

        bump = nodes.new(type='ShaderNodeBump')
        bump.name = "AF_Sculpt_Bump"
        bump.location = (-420, -220)
        bump.inputs["Strength"].default_value = min(1.0, 0.22 * GRAIN_WEAR_STRENGTH)
        bump.inputs["Distance"].default_value = 0.015
        links.new(noise.outputs["Fac"], bump.inputs["Height"])

        # Conectar Normal existente al Bump si ya tenía Normal Map del .GLB
        if bsdf.inputs["Normal"].is_linked:
            old_normal_socket = bsdf.inputs["Normal"].links[0].from_socket
            links.new(old_normal_socket, bump.inputs["Normal"])
        links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])

        # Potenciar emisión en materiales de ojos, runas, estela de corte y partículas
        if "Emission Strength" in bsdf.inputs and bsdf.inputs["Emission Strength"].default_value > 0.1:
            bsdf.inputs["Emission Strength"].default_value = max(
                bsdf.inputs["Emission Strength"].default_value,
                EMISSIVE_STRENGTH
            )
            if "Emission Color" in bsdf.inputs:
                bsdf.inputs["Emission Color"].default_value = emi_rgba

    print("[AetherForge] Nodos PBR (SSS + Curvature Pointiness + Bump 3D) aplicados con éxito.")


def main():
    ensure_glb_imported()
    apply_sculpt_and_subsurf_modifiers()
    enhance_pbr_materials_with_curvature_and_sss()

    if AUTO_EXPORT_ENHANCED_GLB:
        print(f"[AetherForge] Exportando versión subdividida a {GLB_OUTPUT_PATH}...")
        bpy.ops.export_scene.gltf(
            filepath=GLB_OUTPUT_PATH,
            export_format='GLB',
            export_apply=True,
            export_animations=True,
            export_skins=True
        )
        print("[AetherForge] Exportación completada.")


if __name__ == "__main__":
    main()
`;
}


