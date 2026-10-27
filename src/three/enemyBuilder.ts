import * as THREE from 'three';
import {
  AnimationState,
  EnemyConfig,
  PartInspectionInfo,
  getEffectiveMeshDensity,
  getEffectiveTexturePattern,
} from '../types/enemy';
import { ProceduralTextureSet } from './proceduralTextures';

// Pre-allocated scratch helpers for zero-GC in the animation loop
const _targetPos = new THREE.Vector3();
const _dummyObj = new THREE.Object3D();
const _shardColor = new THREE.Color();

export interface BuiltEnemyRig {
  rootGroup: THREE.Group;
  exportableGroup: THREE.Group; // Clean enemy hierarchy for .GLB export (no gizmos)
  hitboxGroup: THREE.Group;     // Toggleable hitboxes, aggro rings & socket axes
  materials: {
    primaryArmorMat: THREE.MeshPhysicalMaterial;
    secondaryTrimMat: THREE.MeshPhysicalMaterial;
    underbodyMat: THREE.MeshStandardMaterial;
    emissiveRuneMat: THREE.MeshStandardMaterial;
    crystalCoreMat: THREE.MeshPhysicalMaterial;
  };
  parts: {
    pelvisGroup: THREE.Group;
    torsoGroup: THREE.Group;
    neckGroup: THREE.Group;
    headGroup: THREE.Group;
    jawGroup: THREE.Group;
    extraHeads: THREE.Group[];
    leftShoulderGroup: THREE.Group;
    rightShoulderGroup: THREE.Group;
    leftArmGroup: THREE.Group;
    rightArmGroup: THREE.Group;
    leftLegGroup: THREE.Group;
    rightLegGroup: THREE.Group;
    leftWingGroup: THREE.Group;
    rightWingGroup: THREE.Group;
    tailGroup: THREE.Group;
    dorsalGroup: THREE.Group;
    weaponGroup: THREE.Group;
    offhandGroup: THREE.Group;
    coreMesh: THREE.Mesh;
    breathConeMesh: THREE.Mesh;
    auraInstancedMesh: THREE.InstancedMesh | null;
  };
  sockets: {
    head: THREE.Object3D;
    weaponR: THREE.Object3D;
    shieldL: THREE.Object3D;
    core: THREE.Object3D;
  };
  explodableParts: THREE.Object3D[];
  shardData: Array<{
    angle: number;
    radius: number;
    yOffset: number;
    speed: number;
    scale: number;
  }>;
  metrics: {
    totalVertices: number;
    totalTriangles: number;
    partCount: number;
  };
  dispose: () => void;
}

function tagExplodablePart(
  obj: THREE.Object3D,
  explodeDir: THREE.Vector3,
  partInfo: PartInspectionInfo
) {
  obj.userData.basePos = obj.position.clone();
  obj.userData.explodeVec = explodeDir.clone();
  obj.userData.partInfo = partInfo;
  obj.traverse((child) => {
    if ((child as THREE.Mesh).isMesh) {
      child.userData.partInfo = partInfo;
      child.layers.enable(1);
    }
  });
}

function applySurfaceTexturesToMaterials(
  materials: BuiltEnemyRig['materials'],
  config: EnemyConfig,
  textures: ProceduralTextureSet
) {
  const patternId = getEffectiveTexturePattern(config);
  const relief = config.materials.textureRelief ?? 0.75;
  const surf = patternId !== 'none' ? textures.patterns[patternId] : null;

  const newAlbedo = surf ? surf.albedoMap : null;
  const newBump = surf ? surf.bumpMap : textures.armorBumpMap;

  if (materials.primaryArmorMat.map !== newAlbedo) {
    materials.primaryArmorMat.map = newAlbedo;
    materials.primaryArmorMat.needsUpdate = true;
  }
  materials.primaryArmorMat.bumpMap = newBump;
  materials.primaryArmorMat.bumpScale = 0.032 * relief;

  if (materials.secondaryTrimMat.map !== newAlbedo) {
    materials.secondaryTrimMat.map = newAlbedo;
    materials.secondaryTrimMat.needsUpdate = true;
  }
  materials.secondaryTrimMat.bumpMap = newBump;
  materials.secondaryTrimMat.bumpScale = 0.022 * relief;

  if (materials.underbodyMat.map !== newAlbedo) {
    materials.underbodyMat.map = newAlbedo;
    materials.underbodyMat.needsUpdate = true;
  }
  materials.underbodyMat.bumpMap = newBump;
  materials.underbodyMat.bumpScale = 0.028 * relief;
}

/**
 * Updates materials, surface textures, and RPG userData in-place (< 0.1ms) without rebuilding geometry.
 */
export function updateEnemyMaterialsInPlace(
  rig: BuiltEnemyRig,
  config: EnemyConfig,
  textures: ProceduralTextureSet,
  isSoftwareGl: boolean,
  wireframe: boolean
) {
  const { materials: matCfg } = config;
  const {
    primaryArmorMat,
    secondaryTrimMat,
    underbodyMat,
    emissiveRuneMat,
    crystalCoreMat,
  } = rig.materials;

  primaryArmorMat.color.set(matCfg.primaryColor);
  primaryArmorMat.metalness = matCfg.metalness;
  primaryArmorMat.roughness = matCfg.roughness;
  primaryArmorMat.anisotropy = isSoftwareGl ? 0 : matCfg.anisotropy;
  primaryArmorMat.wireframe = wireframe;

  secondaryTrimMat.color.set(matCfg.secondaryColor);
  secondaryTrimMat.metalness = Math.min(1, matCfg.metalness + 0.12);
  secondaryTrimMat.roughness = Math.max(0.08, (matCfg.roughness + 0.32) % 0.85);
  secondaryTrimMat.wireframe = wireframe;

  underbodyMat.color.set(matCfg.underbodyColor);
  underbodyMat.metalness = Math.max(0.04, matCfg.metalness * 0.22);
  underbodyMat.roughness = Math.min(0.92, matCfg.roughness + 0.42);
  underbodyMat.wireframe = wireframe;

  emissiveRuneMat.color.set(matCfg.emissiveColor);
  emissiveRuneMat.emissive.set(matCfg.emissiveColor);
  emissiveRuneMat.emissiveIntensity = matCfg.emissiveIntensity;
  emissiveRuneMat.wireframe = wireframe;

  crystalCoreMat.color.set(matCfg.emissiveColor).lerp(new THREE.Color('#ffffff'), 0.35);
  crystalCoreMat.emissive.set(matCfg.emissiveColor);
  crystalCoreMat.emissiveIntensity = matCfg.emissiveIntensity * 0.65;
  crystalCoreMat.wireframe = wireframe;

  applySurfaceTexturesToMaterials(rig.materials, config, textures);

  rig.exportableGroup.userData.name = config.name;
  rig.exportableGroup.userData.title = config.title;
  rig.exportableGroup.userData.rpgStats = { ...config.stats };
}

export function buildProceduralEnemy(
  config: EnemyConfig,
  textures: ProceduralTextureSet,
  isSoftwareGl: boolean,
  wireframe: boolean
): BuiltEnemyRig {
  const rootGroup = new THREE.Group();
  rootGroup.name = `${config.id}_root`;

  const exportableGroup = new THREE.Group();
  exportableGroup.name = config.id;
  exportableGroup.userData = {
    name: config.name,
    title: config.title,
    raceCategory: config.raceCategory,
    bodyPlan: config.bodyPlan,
    archetype: config.archetype,
    rpgStats: { ...config.stats },
    sockets: ['socket_head', 'socket_weapon_r', 'socket_shield_l', 'socket_core'],
    generator: 'AetherForge 3D RPG Studio',
  };
  rootGroup.add(exportableGroup);

  const geometriesToDispose: THREE.BufferGeometry[] = [];
  const materialsToDispose: THREE.Material[] = [];

  const trackGeo = <T extends THREE.BufferGeometry>(g: T): T => {
    geometriesToDispose.push(g);
    return g;
  };
  const trackMat = <T extends THREE.Material>(m: T): T => {
    materialsToDispose.push(m);
    return m;
  };

  // Safe defaults for every morphology parameter so randomizer or imports never produce NaN
  const rawM = config.morphology;
  const m = {
    heightScale: Number.isFinite(rawM.heightScale) ? rawM.heightScale : 1.0,
    torsoBulk: Number.isFinite(rawM.torsoBulk) ? rawM.torsoBulk : 1.0,
    shoulderWidth: Number.isFinite(rawM.shoulderWidth) ? rawM.shoulderWidth : 1.0,
    hornLength: Number.isFinite(rawM.hornLength) ? rawM.hornLength : 0.8,
    hornCurve: Number.isFinite(rawM.hornCurve) ? rawM.hornCurve : 0.6,
    pauldronSize: Number.isFinite(rawM.pauldronSize) ? rawM.pauldronSize : 1.0,
    weaponScale: Number.isFinite(rawM.weaponScale) ? rawM.weaponScale : 1.0,
    dorsalScale: Number.isFinite(rawM.dorsalScale) ? rawM.dorsalScale : 1.0,
    limbLength: Number.isFinite(rawM.limbLength) ? rawM.limbLength : 1.0,
    neckLength: Number.isFinite(rawM.neckLength) ? rawM.neckLength : 0.7,
    tailLength: Number.isFinite(rawM.tailLength) ? rawM.tailLength : 0.8,
    snoutLength: Number.isFinite(rawM.snoutLength) ? rawM.snoutLength : 0.8,
    headCount: Number.isFinite(rawM.headCount) ? rawM.headCount : 1,
    auraShardCount: Number.isFinite(rawM.auraShardCount) ? rawM.auraShardCount : 12,
  };

  // Polygon Density (LOD) Multiplier
  const density = getEffectiveMeshDensity(config);
  const lodMult = density === 'low' ? 0.55 : density === 'high' ? 1.55 : 1.0;
  const seg = (base: number, min = 4) => Math.max(min, Math.round(base * lodMult));

  const { materials: matCfg, bodyPlan } = config;
  const isQuadruped = bodyPlan === 'quadruped_beast' || bodyPlan === 'dragon_wyvern';
  const isDragon = bodyPlan === 'dragon_wyvern';
  const isArachnid = bodyPlan === 'arachnid_scorpion';
  const isSerpentine = bodyPlan === 'serpentine_naga';
  const isFloating = bodyPlan === 'floating_magical';
  const isCentaur = config.archetype === 'astral_centaur';
  const isBeholder = config.archetype === 'arcane_beholder';

  // --- STUDIO PBR MATERIALS WITH PROCEDURAL ALBEDO + BUMP TEXTURES ---
  const primaryArmorMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_PrimarySurface',
      color: new THREE.Color(matCfg.primaryColor),
      metalness: matCfg.metalness,
      roughness: matCfg.roughness,
      roughnessMap: textures.armorRoughnessMap,
      anisotropy: isSoftwareGl ? 0 : matCfg.anisotropy,
      anisotropyRotation: Math.PI / 4,
      clearcoat: isDragon ? 0.42 : 0.2,
      clearcoatRoughness: 0.26,
      wireframe,
    })
  );

  const secondaryTrimMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_SecondaryScalesOrTrim',
      color: new THREE.Color(matCfg.secondaryColor),
      metalness: Math.min(1, matCfg.metalness + 0.12),
      roughness: Math.max(0.08, (matCfg.roughness + 0.32) % 0.85),
      anisotropy: isSoftwareGl ? 0 : matCfg.anisotropy * 0.8,
      wireframe,
    })
  );

  const underbodyMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_UnderbodyHide',
      color: new THREE.Color(matCfg.underbodyColor),
      metalness: Math.max(0.04, matCfg.metalness * 0.22),
      roughness: Math.min(0.92, matCfg.roughness + 0.42),
      wireframe,
    })
  );

  const ivoryFangMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_IvoryFangsAndHorns',
      color: new THREE.Color('#e8e1ce'),
      metalness: 0.08,
      roughness: 0.38,
      bumpMap: textures.armorBumpMap,
      bumpScale: 0.008,
      wireframe,
    })
  );

  const bladeAlloyMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_WeaponBlade',
      color: new THREE.Color('#dce5f2'),
      metalness: 0.94,
      roughness: 0.14,
      map: textures.patterns.damascus_plate.albedoMap,
      bumpMap: textures.patterns.damascus_plate.bumpMap,
      bumpScale: 0.015,
      anisotropy: isSoftwareGl ? 0 : 0.75,
      clearcoat: 0.4,
      wireframe,
    })
  );

  const emissiveRuneMat = trackMat(
    new THREE.MeshStandardMaterial({
      name: 'Mat_RunicEmissive',
      color: new THREE.Color(matCfg.emissiveColor),
      emissive: new THREE.Color(matCfg.emissiveColor),
      emissiveIntensity: matCfg.emissiveIntensity,
      roughness: 0.15,
      metalness: 0.1,
      wireframe,
    })
  );

  const crystalCoreMat = trackMat(
    new THREE.MeshPhysicalMaterial({
      name: 'Mat_ArcaneCrystal',
      color: new THREE.Color(matCfg.emissiveColor).lerp(new THREE.Color('#ffffff'), 0.35),
      emissive: new THREE.Color(matCfg.emissiveColor),
      emissiveIntensity: matCfg.emissiveIntensity * 0.65,
      metalness: 0.1,
      roughness: 0.08,
      transmission: isSoftwareGl ? 0 : matCfg.crystalTransmission,
      transparent: true,
      opacity: isSoftwareGl ? 0.78 : 1.0,
      ior: 1.75,
      thickness: 1.2,
      clearcoat: 1.0,
      wireframe,
    })
  );

  applySurfaceTexturesToMaterials(
    {
      primaryArmorMat,
      secondaryTrimMat,
      underbodyMat,
      emissiveRuneMat,
      crystalCoreMat,
    },
    config,
    textures
  );

  const explodableParts: THREE.Object3D[] = [];

  const legHeight = isSerpentine
    ? 0.55 * m.limbLength
    : isArachnid
    ? 0.68 * m.limbLength
    : isQuadruped
    ? 0.92 * m.limbLength
    : 1.05 * m.limbLength;
  const pelvisY = isFloating ? legHeight + 0.38 : legHeight;

  // 1. PELVIS / HAUNCHES / SERPENT COIL WITH ANATOMICAL CONTINUITY
  const pelvisGroup = new THREE.Group();
  pelvisGroup.name = 'Pelvis_Chassis';
  pelvisGroup.position.set(0, pelvisY, isQuadruped ? -0.35 : 0);
  exportableGroup.add(pelvisGroup);

  if (isSerpentine) {
    // Continuous Coiled Serpentine / Naga / Hydra Body sitting flush at Y = 0
    const coilCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, 0.22, 0),
      new THREE.Vector3(0.08 * m.torsoBulk, -0.08, 0.08),
      new THREE.Vector3(0.32 * m.torsoBulk, -legHeight * 0.45, 0.25),
      new THREE.Vector3(0, -legHeight + 0.14, 0.48),
      new THREE.Vector3(-0.42 * m.torsoBulk, -legHeight + 0.12, 0.05),
      new THREE.Vector3(-0.15 * m.torsoBulk, -legHeight + 0.1, -0.55),
      new THREE.Vector3(0.35 * m.torsoBulk, -legHeight + 0.08, -0.88),
      new THREE.Vector3(0.12 * m.torsoBulk, -legHeight + 0.05, -1.18),
    ]);
    const coilGeo = trackGeo(
      new THREE.TubeGeometry(coilCurve, seg(22, 12), 0.23 * m.torsoBulk, seg(10, 6), false)
    );
    const coilMesh = new THREE.Mesh(coilGeo, primaryArmorMat);
    coilMesh.castShadow = true;
    coilMesh.receiveShadow = true;
    pelvisGroup.add(coilMesh);

    // Waist transition collar fusing upper torso into serpentine coil
    const waistCollarPts = [
      new THREE.Vector2(0.24 * m.torsoBulk, 0.0),
      new THREE.Vector2(0.31 * m.torsoBulk, 0.14),
      new THREE.Vector2(0.28 * m.torsoBulk, 0.28),
    ];
    const waistCollarGeo = trackGeo(new THREE.LatheGeometry(waistCollarPts, seg(10, 6)));
    const waistCollarMesh = new THREE.Mesh(waistCollarGeo, secondaryTrimMat);
    waistCollarMesh.position.y = 0.04;
    pelvisGroup.add(waistCollarMesh);
  } else if (isQuadruped) {
    // Sculpted Organic Ribcage + Abdomen Barrel with closed rounded ends
    const barrelPts = [
      new THREE.Vector2(0.01, -0.68),
      new THREE.Vector2(0.28 * m.torsoBulk, -0.55),
      new THREE.Vector2(0.42 * m.torsoBulk, -0.15),
      new THREE.Vector2(0.49 * m.torsoBulk, 0.38),
      new THREE.Vector2(0.41 * m.torsoBulk, 0.78),
      new THREE.Vector2(0.01, 0.92),
    ];
    const barrelGeo = trackGeo(new THREE.LatheGeometry(barrelPts, seg(12, 6)));
    barrelGeo.rotateX(Math.PI / 2);
    const barrelMesh = new THREE.Mesh(barrelGeo, primaryArmorMat);
    barrelMesh.position.set(0, 0.14, 0.22);
    barrelMesh.castShadow = true;
    pelvisGroup.add(barrelMesh);

    // Continuous Ventral Belly Scutes (Segmented Underbelly)
    const bellyRingCount = seg(6, 4);
    for (let b = 0; b < bellyRingCount; b++) {
      const frac = b / Math.max(1, bellyRingCount - 1);
      const zOff = -0.35 + frac * 1.05;
      const rad = (0.31 + Math.sin(frac * Math.PI) * 0.12) * m.torsoBulk;
      const scuteGeo = trackGeo(new THREE.SphereGeometry(rad, seg(8, 5), seg(6, 4)));
      scuteGeo.scale(0.88, 0.55, 0.55);
      const scuteMesh = new THREE.Mesh(scuteGeo, secondaryTrimMat);
      scuteMesh.position.set(0, -0.04, zOff);
      pelvisGroup.add(scuteMesh);
    }

    // Continuous Dorsal Vertebral Ridge along the quadruped back
    const spineCount = seg(7, 4);
    for (let s = 0; s < spineCount; s++) {
      const frac = s / Math.max(1, spineCount - 1);
      const zOff = -0.42 + frac * 1.05;
      const ridgeGeo = trackGeo(new THREE.ConeGeometry(0.055 * m.torsoBulk, 0.18, 4));
      ridgeGeo.rotateX(-0.35);
      const ridgeMesh = new THREE.Mesh(ridgeGeo, secondaryTrimMat);
      ridgeMesh.position.set(0, 0.52 * m.torsoBulk, zOff);
      pelvisGroup.add(ridgeMesh);
    }
  } else if (isArachnid) {
    // Segmented Opisthosoma (Abdomen) + Fused Cephalothorax Bridge
    const abdomenGeo = trackGeo(new THREE.SphereGeometry(0.48 * m.torsoBulk, seg(12, 6), seg(10, 5)));
    abdomenGeo.scale(1.05, 0.74, 1.48);
    const abdomenMesh = new THREE.Mesh(abdomenGeo, primaryArmorMat);
    abdomenMesh.position.set(0, 0.1, -0.38);
    abdomenMesh.castShadow = true;
    pelvisGroup.add(abdomenMesh);

    // Chitinous Tergite Rings across the abdomen for segmented continuity
    for (let r = 0; r < 3; r++) {
      const archGeo = trackGeo(
        new THREE.TorusGeometry((0.42 - r * 0.05) * m.torsoBulk, 0.035, 5, seg(10, 6), Math.PI)
      );
      const archMesh = new THREE.Mesh(archGeo, secondaryTrimMat);
      archMesh.position.set(0, 0.12, -0.15 - r * 0.24);
      archMesh.rotation.x = -0.2;
      pelvisGroup.add(archMesh);
    }

    // Fused Cephalothorax Bridge connecting abdomen to front torso & leg sockets
    const cephaloGeo = trackGeo(new THREE.SphereGeometry(0.36 * m.torsoBulk, seg(10, 6), seg(8, 5)));
    cephaloGeo.scale(1.15, 0.65, 1.25);
    const cephaloMesh = new THREE.Mesh(cephaloGeo, secondaryTrimMat);
    cephaloMesh.position.set(0, 0.06, 0.12);
    pelvisGroup.add(cephaloMesh);
  } else {
    // Biped / Floating Pelvis + Sculpted Lumbar-Abdominal Bridge (No torso gap!)
    const pelvisPts = [
      new THREE.Vector2(0.24 * m.torsoBulk, -0.18),
      new THREE.Vector2(0.36 * m.torsoBulk, 0.0),
      new THREE.Vector2(0.31 * m.torsoBulk, 0.18),
      new THREE.Vector2(0.34 * m.torsoBulk, 0.34),
    ];
    const pelvisGeo = trackGeo(new THREE.LatheGeometry(pelvisPts, seg(10, 6)));
    pelvisGeo.scale(1.12, 1.0, 0.85);
    const pelvisMesh = new THREE.Mesh(pelvisGeo, underbodyMat);
    pelvisMesh.castShadow = true;
    pelvisGroup.add(pelvisMesh);

    // Segmented Waist Cincture Rings bridging pelvis into upper chest
    for (let w = 0; w < 2; w++) {
      const cinctureGeo = trackGeo(
        new THREE.TorusGeometry((0.31 + w * 0.02) * m.torsoBulk, 0.036, 5, seg(12, 6))
      );
      cinctureGeo.rotateX(Math.PI / 2);
      cinctureGeo.scale(1.12, 0.85, 1.0);
      const cinctureMesh = new THREE.Mesh(cinctureGeo, secondaryTrimMat);
      cinctureMesh.position.y = 0.08 + w * 0.11;
      pelvisGroup.add(cinctureMesh);
    }

    // Front Armored Fauld / Tasset
    const beltShape = new THREE.Shape();
    beltShape.moveTo(-0.26 * m.torsoBulk, 0.12);
    beltShape.lineTo(0.26 * m.torsoBulk, 0.12);
    beltShape.lineTo(0.18 * m.torsoBulk, -0.26);
    beltShape.lineTo(0, -0.36);
    beltShape.lineTo(-0.18 * m.torsoBulk, -0.26);
    beltShape.closePath();

    const beltGeo = trackGeo(
      new THREE.ExtrudeGeometry(beltShape, {
        depth: 0.08,
        bevelEnabled: true,
        bevelThickness: 0.02,
        bevelSize: 0.02,
        bevelSegments: 1,
      })
    );
    const frontTasset = new THREE.Mesh(beltGeo, secondaryTrimMat);
    frontTasset.position.set(0, 0.02, 0.25 * m.torsoBulk);
    frontTasset.castShadow = true;
    pelvisGroup.add(frontTasset);

    // Lateral Hip Guards (Tassets) bridging pelvis to upper thighs
    for (const side of [-1, 1]) {
      const sideTassetGeo = trackGeo(
        new THREE.BoxGeometry(0.1 * m.torsoBulk, 0.28, 0.28 * m.torsoBulk)
      );
      const sideTasset = new THREE.Mesh(sideTassetGeo, secondaryTrimMat);
      sideTasset.position.set(side * 0.34 * m.torsoBulk, -0.06, 0);
      sideTasset.rotation.z = side * 0.22;
      pelvisGroup.add(sideTasset);
    }

    if (
      config.archetype === 'arch_lich' ||
      config.archetype === 'solar_phoenix' ||
      config.archetype === 'void_djinn'
    ) {
      const robePts = [
        new THREE.Vector2(0.26 * m.torsoBulk, 0.08),
        new THREE.Vector2(0.38 * m.torsoBulk, -0.45),
        new THREE.Vector2(0.54 * m.torsoBulk, -0.95),
      ];
      const robeGeo = trackGeo(new THREE.LatheGeometry(robePts, seg(12, 6)));
      const robeMesh = new THREE.Mesh(robeGeo, primaryArmorMat);
      robeMesh.castShadow = true;
      pelvisGroup.add(robeMesh);
    }
  }

  tagExplodablePart(pelvisGroup, new THREE.Vector3(0, -0.35, -0.25), {
    name: isSerpentine
      ? 'Anillos Serpentinos Inferiores'
      : isQuadruped
      ? 'Chasis Corporal y Placas Ventrales'
      : isArachnid
      ? 'Abdomen y Cefalotórax Quitinoso'
      : 'Pelvis, Abdomen y Faldón Acorazado',
    category: 'Primary Silhouette',
    vertexCount: 280,
    triangleCount: 220,
    materialSummary: 'Estructura continua con textura procedural',
  });
  explodableParts.push(pelvisGroup);

  // 2. CONTINUOUS ARTICULATED LEGS (Hip Haunch -> Thigh -> Knee Poleyn -> Calf -> Ankle -> Clawed Foot)
  const buildLeg = (isLeft: boolean): THREE.Group => {
    const side = isLeft ? 1 : -1;
    const legGroup = new THREE.Group();
    legGroup.name = isLeft ? 'Leg_L' : 'Leg_R';
    const zPos = isQuadruped ? -0.48 : 0;
    legGroup.position.set(side * 0.28 * m.torsoBulk, pelvisY, zPos);
    exportableGroup.add(legGroup);

    if (isSerpentine) {
      return legGroup;
    }

    if (isFloating && isBeholder) {
      const tendrilCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(0, 0.08, 0),
        new THREE.Vector3(side * 0.12, -0.35, -0.1),
        new THREE.Vector3(side * 0.05, -0.7, 0.08),
      ]);
      const tendrilGeo = trackGeo(
        new THREE.TubeGeometry(tendrilCurve, seg(8, 5), 0.048, seg(6, 4), false)
      );
      legGroup.add(new THREE.Mesh(tendrilGeo, underbodyMat));
    } else if (isArachnid) {
      for (let k = 0; k < 4; k++) {
        const zOff = (k - 1.5) * 0.28;
        const reachX = side * (0.82 + (k % 2) * 0.16) * m.limbLength;

        // Coxa Hip Socket welding each spider leg into the cephalothorax
        const coxaGeo = trackGeo(new THREE.SphereGeometry(0.085 * m.torsoBulk, seg(6, 4), seg(5, 4)));
        const coxaMesh = new THREE.Mesh(coxaGeo, secondaryTrimMat);
        coxaMesh.position.set(0, 0.04, zOff * 0.52);
        legGroup.add(coxaMesh);

        const curve = new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0.04, zOff * 0.52),
          new THREE.Vector3(reachX * 0.55, 0.36, zOff * 0.9),
          new THREE.Vector3(reachX * 1.05, -legHeight * 0.35, zOff * 1.25),
          new THREE.Vector3(reachX * 0.92, -legHeight + 0.02, zOff * 1.35),
        ]);
        const spiderLegGeo = trackGeo(
          new THREE.TubeGeometry(curve, seg(10, 6), 0.056, seg(6, 4), false)
        );
        const spiderMesh = new THREE.Mesh(spiderLegGeo, primaryArmorMat);
        spiderMesh.castShadow = true;
        legGroup.add(spiderMesh);

        // Chitin Knee Joint Node at apex of spider leg
        const apexJointGeo = trackGeo(new THREE.SphereGeometry(0.072, seg(6, 4), seg(5, 4)));
        const apexJoint = new THREE.Mesh(apexJointGeo, secondaryTrimMat);
        apexJoint.position.set(reachX * 0.55, 0.36, zOff * 0.9);
        legGroup.add(apexJoint);
      }
    } else {
      // 1) Gluteal / Hip Socket Haunch welding leg seamlessly into pelvis
      const hipHaunchGeo = trackGeo(
        new THREE.SphereGeometry(0.21 * m.torsoBulk, seg(8, 5), seg(6, 4))
      );
      hipHaunchGeo.scale(0.95, 1.18, 1.15);
      const hipHaunchMesh = new THREE.Mesh(
        hipHaunchGeo,
        isQuadruped ? primaryArmorMat : secondaryTrimMat
      );
      hipHaunchMesh.position.set(-side * 0.04, 0.02, isQuadruped ? 0.05 : 0);
      legGroup.add(hipHaunchMesh);

      // 2) Sculpted Organic Thigh
      const thighPts = [
        new THREE.Vector2(0.13 * m.torsoBulk, 0),
        new THREE.Vector2(0.19 * m.torsoBulk, legHeight * 0.28),
        new THREE.Vector2(0.17 * m.torsoBulk, legHeight * 0.52),
      ];
      const thighGeo = trackGeo(new THREE.LatheGeometry(thighPts, seg(8, 5)));
      const thighMesh = new THREE.Mesh(thighGeo, isQuadruped ? primaryArmorMat : underbodyMat);
      thighMesh.position.set(0, -legHeight * 0.48, isQuadruped ? -0.05 : 0);
      thighMesh.rotation.x = isQuadruped ? 0.22 : 0;
      thighMesh.castShadow = true;
      legGroup.add(thighMesh);

      // 3) Articulated Knee Sphere + Armored Knee Poleyn (No gap between thigh and shin!)
      const kneeSphereGeo = trackGeo(
        new THREE.SphereGeometry(0.135 * m.torsoBulk, seg(8, 5), seg(6, 4))
      );
      const kneeSphere = new THREE.Mesh(kneeSphereGeo, underbodyMat);
      kneeSphere.position.set(0, -legHeight * 0.48, 0.02);
      legGroup.add(kneeSphere);

      const kneeGuardGeo = trackGeo(new THREE.OctahedronGeometry(0.14 * m.torsoBulk, 0));
      kneeGuardGeo.scale(1.0, 1.25, 0.85);
      const kneeMesh = new THREE.Mesh(kneeGuardGeo, secondaryTrimMat);
      kneeMesh.position.set(0, -legHeight * 0.48, 0.11);
      legGroup.add(kneeMesh);

      // 4) Sculpted Calf / Greave overlapping from knee down to ankle
      const greavePts = [
        new THREE.Vector2(0.11 * m.torsoBulk, 0),
        new THREE.Vector2(0.165 * m.torsoBulk, legHeight * 0.26),
        new THREE.Vector2(0.135 * m.torsoBulk, legHeight * 0.5),
      ];
      const greaveGeo = trackGeo(new THREE.LatheGeometry(greavePts, seg(8, 5)));
      const greaveMesh = new THREE.Mesh(greaveGeo, primaryArmorMat);
      greaveMesh.position.y = -legHeight * 0.94;
      greaveMesh.castShadow = true;
      legGroup.add(greaveMesh);

      // 5) Ankle Joint + Grounded Sabaton / Clawed Paw at Y = 0
      const ankleGeo = trackGeo(
        new THREE.SphereGeometry(0.115 * m.torsoBulk, seg(6, 4), seg(5, 4))
      );
      const ankleMesh = new THREE.Mesh(ankleGeo, secondaryTrimMat);
      ankleMesh.position.set(0, -legHeight + 0.11, 0.02);
      legGroup.add(ankleMesh);

      const footGeo = trackGeo(new THREE.BoxGeometry(0.24 * m.torsoBulk, 0.14, 0.38));
      const footMesh = new THREE.Mesh(footGeo, secondaryTrimMat);
      footMesh.position.set(0, -legHeight + 0.07, 0.08);
      legGroup.add(footMesh);

      if (isDragon || isQuadruped || config.archetype === 'labyrinth_minotaur') {
        for (let c = -1; c <= 1; c++) {
          const clawGeo = trackGeo(new THREE.ConeGeometry(0.034, 0.19, 4));
          clawGeo.rotateX(Math.PI / 2);
          const clawMesh = new THREE.Mesh(clawGeo, ivoryFangMat);
          clawMesh.position.set(c * 0.07, -legHeight + 0.05, 0.3);
          legGroup.add(clawMesh);
        }
      }
    }

    tagExplodablePart(legGroup, new THREE.Vector3(side * 0.5, -0.45, -0.15), {
      name: isLeft ? 'Extremidad Inferior Izquierda' : 'Extremidad Inferior Derecha',
      category: 'Primary Silhouette',
      vertexCount: 310,
      triangleCount: 250,
      materialSummary: 'Cadera, muslo, rodillera, greba y garra continua en Y=0',
    });
    explodableParts.push(legGroup);

    return legGroup;
  };

  const leftLegGroup = buildLeg(true);
  const rightLegGroup = buildLeg(false);

  // 3. TORSO / DRAGON CHEST / BEHOLDER ORB WITH GORGET & STERNUM CONTINUITY
  const torsoGroup = new THREE.Group();
  torsoGroup.name = 'Torso_Main';
  if (isDragon) {
    torsoGroup.position.set(0, 0.22, 0.44);
    torsoGroup.rotation.x = 0.36;
  } else if (isQuadruped && !isCentaur) {
    torsoGroup.position.set(0, 0.16, 0.46);
    torsoGroup.rotation.x = 0.48;
  } else if (isCentaur) {
    torsoGroup.position.set(0, 0.24, 0.54);
  } else {
    torsoGroup.position.set(0, 0.15, 0);
  }
  pelvisGroup.add(torsoGroup);

  const torsoHeight = 0.92 * m.heightScale;

  if (isBeholder) {
    const orbBodyGeo = trackGeo(
      new THREE.IcosahedronGeometry(0.62 * m.torsoBulk, density === 'low' ? 1 : 2)
    );
    const orbBodyMesh = new THREE.Mesh(orbBodyGeo, primaryArmorMat);
    orbBodyMesh.position.y = torsoHeight * 0.55;
    orbBodyMesh.castShadow = true;
    torsoGroup.add(orbBodyMesh);

    const browGeo = trackGeo(new THREE.TorusGeometry(0.44 * m.torsoBulk, 0.08, 6, seg(18, 10)));
    const browMesh = new THREE.Mesh(browGeo, secondaryTrimMat);
    browMesh.position.set(0, torsoHeight * 0.58, 0.42 * m.torsoBulk);
    torsoGroup.add(browMesh);
  } else {
    // Continuous Torso Lathe starting at y = -0.08 to overlap cleanly into pelvis/barrel
    const cuirassPts = [
      new THREE.Vector2(0.22 * m.torsoBulk, -0.08),
      new THREE.Vector2(0.32 * m.torsoBulk, 0.05),
      new THREE.Vector2(0.42 * m.torsoBulk, torsoHeight * 0.38),
      new THREE.Vector2(0.52 * m.torsoBulk * m.shoulderWidth * 0.85, torsoHeight * 0.78),
      new THREE.Vector2(0.34 * m.torsoBulk, torsoHeight * 0.96),
      new THREE.Vector2(0.22 * m.torsoBulk, torsoHeight * 1.04),
    ];
    const cuirassGeo = trackGeo(new THREE.LatheGeometry(cuirassPts, seg(12, 6)));
    cuirassGeo.scale(1.15, 1.0, 0.84);
    const cuirassMesh = new THREE.Mesh(cuirassGeo, primaryArmorMat);
    cuirassMesh.castShadow = true;
    torsoGroup.add(cuirassMesh);

    // Sculpted Trapezius & Gorget Collar (or Fur Mane Ruff on starter animals)
    const isFeralAnimal = config.raceCategory === 'starter_animals';
    if (isFeralAnimal) {
      const maneGeo = trackGeo(
        new THREE.SphereGeometry(0.36 * m.torsoBulk, seg(10, 6), seg(8, 5))
      );
      maneGeo.scale(1.12, 0.95, 1.18);
      const maneMesh = new THREE.Mesh(maneGeo, secondaryTrimMat);
      maneMesh.position.set(0, torsoHeight * 0.72, 0.14 * m.torsoBulk);
      maneMesh.castShadow = true;
      torsoGroup.add(maneMesh);
    } else {
      const gorgetGeo = trackGeo(
        new THREE.TorusGeometry(0.26 * m.torsoBulk, 0.065 * m.torsoBulk, 6, seg(12, 6))
      );
      gorgetGeo.rotateX(Math.PI / 2 - 0.15);
      const gorgetMesh = new THREE.Mesh(gorgetGeo, secondaryTrimMat);
      gorgetMesh.position.set(0, torsoHeight * 0.93, 0.03);
      torsoGroup.add(gorgetMesh);

      // Front Pectoral Sternum / Plackart Plate
      const chestShape = new THREE.Shape();
      chestShape.moveTo(-0.28 * m.torsoBulk, torsoHeight * 0.78);
      chestShape.lineTo(0.28 * m.torsoBulk, torsoHeight * 0.78);
      chestShape.lineTo(0.2 * m.torsoBulk, torsoHeight * 0.22);
      chestShape.lineTo(0, torsoHeight * 0.04);
      chestShape.lineTo(-0.2 * m.torsoBulk, torsoHeight * 0.22);
      chestShape.closePath();

      const chestPlateGeo = trackGeo(
        new THREE.ExtrudeGeometry(chestShape, {
          depth: 0.13 * m.torsoBulk,
          bevelEnabled: true,
          bevelThickness: 0.025,
          bevelSize: 0.02,
          bevelSegments: 1,
        })
      );
      const chestPlateMesh = new THREE.Mesh(chestPlateGeo, secondaryTrimMat);
      chestPlateMesh.position.set(0, 0, 0.26 * m.torsoBulk);
      chestPlateMesh.castShadow = true;
      torsoGroup.add(chestPlateMesh);
    }
  }

  const socketCore = new THREE.Object3D();
  socketCore.name = 'socket_core';
  socketCore.position.set(0, torsoHeight * 0.56, (isBeholder ? 0.55 : 0.44) * m.torsoBulk);
  torsoGroup.add(socketCore);

  const coreGeo = trackGeo(
    new THREE.OctahedronGeometry(
      (isBeholder ? 0.32 : 0.15) * m.torsoBulk,
      density === 'low' ? 0 : 1
    )
  );
  const coreMesh = new THREE.Mesh(coreGeo, crystalCoreMat);
  coreMesh.name = 'Runic_Core_Crystal';
  coreMesh.position.copy(socketCore.position);
  coreMesh.visible = config.raceCategory !== 'starter_animals';
  torsoGroup.add(coreMesh);

  tagExplodablePart(coreMesh, new THREE.Vector3(0, 0.15, 0.8), {
    name: isBeholder ? 'Iris Central del Ojo Tirano' : 'Núcleo Cristalino Elemental',
    category: 'Accent Detail',
    vertexCount: 128,
    triangleCount: 160,
    materialSummary: 'Cristal PBR con transmisión óptica y emisión',
    socketName: 'socket_core',
  });
  explodableParts.push(coreMesh);

  // 4. CONTINUOUS ARTICULATED NECK & MULTI-HEAD BUILDER (Supports 1, 2, or 3 heads!)
  const neckGroup = new THREE.Group();
  neckGroup.name = 'Neck_Assembly';
  neckGroup.position.set(0, torsoHeight * 0.9, 0.05);
  if (isDragon) {
    neckGroup.rotation.x = -0.34;
  } else if (isQuadruped && !isCentaur) {
    neckGroup.rotation.x = -0.4;
  }
  torsoGroup.add(neckGroup);

  const neckLen = Math.max(0.18, 0.58 * m.neckLength);
  if (isDragon || isSerpentine || m.neckLength > 0.82) {
    // Organic S-curved neck starting inside the gorget (-0.08) so there is zero seam
    const neckCurve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0, -0.08, -0.02),
      new THREE.Vector3(0, neckLen * 0.42, 0.13 * m.neckLength),
      new THREE.Vector3(0, neckLen * 0.82, 0.05 * m.neckLength),
      new THREE.Vector3(0, neckLen * 1.02, 0.17 * m.neckLength),
    ]);
    const neckTubeGeo = trackGeo(
      new THREE.TubeGeometry(neckCurve, seg(10, 6), 0.165 * m.torsoBulk, seg(8, 5), false)
    );
    const neckMesh = new THREE.Mesh(neckTubeGeo, primaryArmorMat);
    neckMesh.castShadow = true;
    neckGroup.add(neckMesh);

    // Cervical Dorsal Spines along the neck curve
    const neckSpineCount = seg(4, 3);
    for (let ns = 1; ns <= neckSpineCount; ns++) {
      const pt = neckCurve.getPoint(ns / (neckSpineCount + 1));
      const nSpineGeo = trackGeo(new THREE.ConeGeometry(0.04, 0.14, 4));
      nSpineGeo.rotateX(-0.6);
      const nSpine = new THREE.Mesh(nSpineGeo, secondaryTrimMat);
      nSpine.position.set(0, pt.y, pt.z - 0.14 * m.torsoBulk);
      neckGroup.add(nSpine);
    }
  } else if (!isBeholder) {
    const shortNeckGeo = trackGeo(
      new THREE.CylinderGeometry(
        0.15 * m.torsoBulk,
        0.21 * m.torsoBulk,
        neckLen * 0.68,
        seg(8, 5)
      )
    );
    const shortNeckMesh = new THREE.Mesh(shortNeckGeo, underbodyMat);
    shortNeckMesh.position.y = neckLen * 0.22;
    neckGroup.add(shortNeckMesh);
  }

  // 5. CONTINUOUS HEAD, SKULL BASE, SNOUT, JAW, HORNS & BREATH EMITTER
  const headGroup = new THREE.Group();
  headGroup.name = 'Head_Helm_Assembly';
  const isLongNeck = isDragon || isSerpentine || m.neckLength > 0.82;
  headGroup.position.set(
    0,
    isBeholder ? -0.1 : isLongNeck ? neckLen * 0.96 : neckLen * 0.42,
    isLongNeck ? 0.16 * m.neckLength : 0.04
  );
  neckGroup.add(headGroup);

  const socketHead = new THREE.Object3D();
  socketHead.name = 'socket_head';
  socketHead.position.set(0, 0.2, 0.28 * m.snoutLength);
  headGroup.add(socketHead);

  const jawGroup = new THREE.Group();
  jawGroup.name = 'Articulated_Lower_Jaw';
  jawGroup.position.set(0, 0.08, 0.08);
  headGroup.add(jawGroup);

  const snoutLen = 0.36 * m.snoutLength;

  if (
    config.helm === 'draconic_snout' ||
    config.helm === 'lupine_maw' ||
    config.helm === 'tauren_bull_horns'
  ) {
    // Sculpted Cranium Sphere + Zygomatic Cheek Plates (Smooth transition from neck to snout!)
    const craniumGeo = trackGeo(new THREE.SphereGeometry(0.24, seg(10, 6), seg(8, 5)));
    craniumGeo.scale(0.95, 0.82, 1.18);
    const craniumMesh = new THREE.Mesh(craniumGeo, primaryArmorMat);
    craniumMesh.position.set(0, 0.19, 0.02);
    craniumMesh.castShadow = true;
    headGroup.add(craniumMesh);

    // Brow & Cheek Ridge Plates bridging cranium into the upper snout
    for (const side of [-1, 1]) {
      const browRidgeGeo = trackGeo(new THREE.BoxGeometry(0.09, 0.08, 0.32));
      const browRidge = new THREE.Mesh(browRidgeGeo, secondaryTrimMat);
      browRidge.position.set(side * 0.14, 0.27, 0.12);
      browRidge.rotation.y = side * 0.18;
      headGroup.add(browRidge);
    }

    // Tapered Upper Snout overlapping smoothly into cranium
    const upperSnoutGeo = trackGeo(
      new THREE.CylinderGeometry(0.09, 0.2, snoutLen + 0.32, seg(8, 5))
    );
    upperSnoutGeo.rotateX(Math.PI / 2);
    upperSnoutGeo.scale(1.0, 0.72, 1.0);
    const upperSnoutMesh = new THREE.Mesh(upperSnoutGeo, primaryArmorMat);
    upperSnoutMesh.position.set(0, 0.17, 0.18 + snoutLen * 0.45);
    headGroup.add(upperSnoutMesh);

    if (config.helm === 'tauren_bull_horns') {
      const noseRingGeo = trackGeo(new THREE.TorusGeometry(0.085, 0.018, 5, seg(10, 6)));
      const noseRingMesh = new THREE.Mesh(noseRingGeo, secondaryTrimMat);
      noseRingMesh.position.set(0, 0.1, 0.36 + snoutLen * 0.62);
      noseRingMesh.rotation.x = 0.3;
      headGroup.add(noseRingMesh);
    }

    for (const side of [-1, 1]) {
      const eyeGeo = trackGeo(new THREE.SphereGeometry(0.044, 6, 6));
      eyeGeo.scale(0.6, 0.8, 1.4);
      const eyeMesh = new THREE.Mesh(eyeGeo, emissiveRuneMat);
      eyeMesh.position.set(side * 0.15, 0.24, 0.16);
      headGroup.add(eyeMesh);
    }

    // Tapered Articulated Lower Jaw + Fang Row
    const lowerJawGeo = trackGeo(
      new THREE.CylinderGeometry(0.07, 0.16, snoutLen + 0.28, seg(6, 4))
    );
    lowerJawGeo.rotateX(Math.PI / 2);
    lowerJawGeo.scale(1.0, 0.5, 1.0);
    const lowerJawMesh = new THREE.Mesh(lowerJawGeo, secondaryTrimMat);
    lowerJawMesh.position.set(0, 0.01, (snoutLen + 0.22) * 0.5);
    jawGroup.add(lowerJawMesh);

    for (const side of [-1, 1]) {
      const isBoar =
        config.archetype === 'rabid_razorback_boar' ||
        config.archetype === 'iron_boar_juggernaut';
      const fangGeo = trackGeo(
        new THREE.ConeGeometry(isBoar ? 0.038 : 0.028, isBoar ? 0.22 : 0.12, 4)
      );
      const fangMesh = new THREE.Mesh(fangGeo, ivoryFangMat);
      fangMesh.position.set(
        side * (isBoar ? 0.11 : 0.08),
        isBoar ? 0.11 : 0.06,
        snoutLen + 0.16
      );
      if (isBoar) {
        fangMesh.rotation.x = 0.35;
        fangMesh.rotation.z = -side * 0.28;
      }
      jawGroup.add(fangMesh);
    }
  } else if (config.helm === 'avian_phoenix_beak') {
    const avianSkullGeo = trackGeo(new THREE.SphereGeometry(0.23, seg(10, 6), seg(8, 5)));
    avianSkullGeo.scale(0.95, 0.9, 1.15);
    const avianSkull = new THREE.Mesh(avianSkullGeo, primaryArmorMat);
    avianSkull.position.y = 0.2;
    headGroup.add(avianSkull);

    const beakGeo = trackGeo(new THREE.ConeGeometry(0.13, snoutLen + 0.34, seg(6, 4)));
    beakGeo.rotateX(Math.PI / 2.1);
    const beakMesh = new THREE.Mesh(beakGeo, secondaryTrimMat);
    beakMesh.position.set(0, 0.18, 0.26);
    headGroup.add(beakMesh);
  } else {
    // Sculpted Greathelm / Crowned Skull with Neck Guard & Visor
    const craniumGeo = trackGeo(
      config.helm === 'monolith_visor'
        ? new THREE.OctahedronGeometry(0.26, density === 'low' ? 0 : 1)
        : new THREE.SphereGeometry(0.23, seg(10, 6), seg(8, 5))
    );
    if (config.helm !== 'monolith_visor') {
      craniumGeo.scale(0.92, 1.12, 1.05);
    }
    const craniumMesh = new THREE.Mesh(craniumGeo, primaryArmorMat);
    craniumMesh.position.y = 0.2;
    headGroup.add(craniumMesh);

    const faceplateGeo = trackGeo(new THREE.CylinderGeometry(0.18, 0.15, 0.28, seg(6, 5)));
    const faceplateMesh = new THREE.Mesh(faceplateGeo, secondaryTrimMat);
    faceplateMesh.position.set(0, 0.15, 0.06);
    headGroup.add(faceplateMesh);

    const visorGeo = trackGeo(new THREE.BoxGeometry(0.24, 0.045, 0.08));
    const visorMesh = new THREE.Mesh(visorGeo, emissiveRuneMat);
    visorMesh.position.set(0, 0.2, 0.18);
    headGroup.add(visorMesh);
  }

  // Curved Horns / Bull Horns / Pointed Lupine & Keltir Ears / Branching Stag Antlers
  if (m.hornLength > 0.05 && !isBeholder) {
    const isLupineEars = config.helm === 'lupine_maw';
    const isStagAntlers = config.archetype === 'furious_antler_stag';

    for (const side of [-1, 1]) {
      const hLen = 0.44 * m.hornLength;
      const hCurve = m.hornCurve;

      if (isLupineEars) {
        // Pointed Feral Ears (Keltir Fox, Wolf, Bear, Panther, Bat)
        const earGeo = trackGeo(new THREE.ConeGeometry(0.075, 0.22 + hLen * 0.45, 4));
        earGeo.scale(0.65, 1.0, 1.25);
        const earMesh = new THREE.Mesh(earGeo, primaryArmorMat);
        earMesh.position.set(side * 0.16, 0.34 + hLen * 0.12, -0.04);
        earMesh.rotation.z = -side * 0.32;
        earMesh.rotation.x = -0.28;
        headGroup.add(earMesh);
      } else {
        const isSweepBack =
          config.helm === 'draconic_snout' || config.helm === 'avian_phoenix_beak';

        // Horn base socket ring welding horn into skull
        const socketRingGeo = trackGeo(new THREE.SphereGeometry(0.062, 6, 5));
        const socketRing = new THREE.Mesh(socketRingGeo, secondaryTrimMat);
        socketRing.position.set(side * 0.15, 0.27, 0.02);
        headGroup.add(socketRing);

        const hornPath = new THREE.CatmullRomCurve3([
          new THREE.Vector3(side * 0.15, 0.27, 0.02),
          new THREE.Vector3(
            side * (0.34 + hLen * 0.45),
            0.36 + hLen * 0.35,
            isSweepBack ? -0.25 * hCurve : 0.08 * hCurve
          ),
          new THREE.Vector3(
            side * (0.26 + hLen * 0.55),
            0.52 + hLen * 0.85,
            isSweepBack ? -0.58 * hCurve : 0.34 * hCurve
          ),
        ]);
        const hornGeo = trackGeo(
          new THREE.TubeGeometry(hornPath, seg(8, 5), 0.045, seg(6, 4), false)
        );
        const hornMat =
          isDragon || config.raceCategory === 'mythological' || isStagAntlers
            ? ivoryFangMat
            : secondaryTrimMat;
        const hornMesh = new THREE.Mesh(hornGeo, hornMat);
        headGroup.add(hornMesh);

        // Secondary branching antler tines for Furious Antler Stag
        if (isStagAntlers) {
          for (let t = 1; t <= 2; t++) {
            const branchGeo = trackGeo(new THREE.ConeGeometry(0.032, 0.26, 4));
            const branchMesh = new THREE.Mesh(branchGeo, ivoryFangMat);
            branchMesh.position.set(
              side * (0.28 + t * 0.12),
              0.48 + t * 0.18,
              0.08 + t * 0.08
            );
            branchMesh.rotation.z = -side * 0.45;
            branchMesh.rotation.x = 0.45;
            headGroup.add(branchMesh);
          }
        }
      }
    }
  }

  // Clone side necks & heads for Multi-Headed Bosses (Hydra, Cerberus, or user slider!)
  const extraHeads: THREE.Group[] = [];
  const headCount = Math.max(1, Math.min(3, Math.round(m.headCount || 1)));
  if (headCount > 1 && !isBeholder) {
    const sides = headCount === 2 ? [-1] : [-1, 1];
    for (const side of sides) {
      const sideNeck = neckGroup.clone(true);
      sideNeck.name = side === 1 ? 'Neck_Head_Left' : 'Neck_Head_Right';
      sideNeck.position.x = side * 0.26 * m.torsoBulk;
      sideNeck.rotation.z = -side * 0.34;
      sideNeck.rotation.y = side * 0.25;
      sideNeck.scale.setScalar(0.88);
      torsoGroup.add(sideNeck);
      extraHeads.push(sideNeck);
    }
  }

  // Elemental Dragon Breath / Arcane Beam Emitter Cone
  const breathGeo = trackGeo(new THREE.ConeGeometry(0.42, 2.2, seg(10, 6), 1, true));
  breathGeo.rotateX(-Math.PI / 2);
  breathGeo.translate(0, 0, 1.15);
  const breathMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: new THREE.Color(matCfg.emissiveColor),
      transparent: true,
      opacity: 0.7,
      side: THREE.DoubleSide,
      depthWrite: false,
      wireframe: true,
    })
  );
  const breathConeMesh = new THREE.Mesh(breathGeo, breathMat);
  breathConeMesh.position.set(0, 0.14, 0.28 + snoutLen);
  breathConeMesh.scale.setScalar(0.001);
  breathConeMesh.visible = false;
  headGroup.add(breathConeMesh);

  tagExplodablePart(headGroup, new THREE.Vector3(0, 0.85, 0.35), {
    name:
      headCount > 1
        ? `Cabezas Múltiples (${headCount}x Hidra/Cerbero)`
        : 'Cráneo, Fauces y Cornamenta Continua',
    category: 'Primary Silhouette',
    vertexCount: 420 * headCount,
    triangleCount: 360 * headCount,
    materialSummary: 'Cráneo esculpido + pómulos + mandíbula móvil + cuernos',
    socketName: 'socket_head',
  });
  explodableParts.push(headGroup);

  // 6. PAULDRONS / SHOULDER ARMOR
  const buildShoulder = (isLeft: boolean): THREE.Group => {
    const side = isLeft ? 1 : -1;
    const shoulderGroup = new THREE.Group();
    shoulderGroup.name = isLeft ? 'Pauldron_L' : 'Pauldron_R';
    shoulderGroup.position.set(
      side * 0.46 * m.torsoBulk * m.shoulderWidth,
      torsoHeight * 0.78,
      0
    );
    torsoGroup.add(shoulderGroup);

    if (!isBeholder) {
      if (config.raceCategory === 'starter_animals' || m.pauldronSize <= 0.42) {
        // Organic Feral Scapular Muscle & Fur Tuft instead of medieval metal armor
        const furShoulderGeo = trackGeo(
          new THREE.SphereGeometry(0.2 * m.torsoBulk, seg(8, 5), seg(6, 4))
        );
        furShoulderGeo.scale(0.95, 1.15, 1.25);
        const furShoulderMesh = new THREE.Mesh(furShoulderGeo, primaryArmorMat);
        furShoulderMesh.castShadow = true;
        shoulderGroup.add(furShoulderMesh);
      } else {
        const pShape = new THREE.Shape();
        pShape.moveTo(-side * 0.06, 0.22 * m.pauldronSize);
        pShape.lineTo(side * 0.36 * m.pauldronSize, 0.08 * m.pauldronSize);
        pShape.lineTo(side * 0.42 * m.pauldronSize, -0.2 * m.pauldronSize);
        pShape.lineTo(-side * 0.04, -0.14 * m.pauldronSize);
        pShape.closePath();

        const pauldronGeo = trackGeo(
          new THREE.ExtrudeGeometry(pShape, {
            depth: 0.34 * m.pauldronSize,
            bevelEnabled: true,
            bevelThickness: 0.025,
            bevelSize: 0.02,
            bevelSegments: 1,
          })
        );
        pauldronGeo.center();
        const pauldronMesh = new THREE.Mesh(pauldronGeo, secondaryTrimMat);
        pauldronMesh.castShadow = true;
        shoulderGroup.add(pauldronMesh);
      }
    }

    tagExplodablePart(shoulderGroup, new THREE.Vector3(side * 0.75, 0.45, 0), {
      name: isLeft ? 'Hombrera / Escama Escapular Izq.' : 'Hombrera / Escama Escapular Der.',
      category: 'Secondary Read',
      vertexCount: 220,
      triangleCount: 180,
      materialSummary: 'Placa escapular superpuesta al deltoides',
    });
    explodableParts.push(shoulderGroup);

    return shoulderGroup;
  };

  const leftShoulderGroup = buildShoulder(true);
  const rightShoulderGroup = buildShoulder(false);

  // 7. CONTINUOUS ARMS / FORE-LEGS (Deltoid Socket -> Bicep -> Elbow Couter -> Vambrace -> Gauntlet)
  const buildArm = (isLeft: boolean): { armGroup: THREE.Group; handSocket: THREE.Object3D } => {
    const side = isLeft ? 1 : -1;
    const armGroup = new THREE.Group();
    armGroup.name = isLeft ? 'Arm_Or_Foreleg_L' : 'Arm_Or_Foreleg_R';

    if (isQuadruped && !isCentaur) {
      armGroup.position.set(side * 0.32 * m.torsoBulk, pelvisY, 0.42);
      exportableGroup.add(armGroup);
    } else {
      armGroup.position.set(
        side * 0.46 * m.torsoBulk * m.shoulderWidth,
        torsoHeight * 0.74,
        0
      );
      torsoGroup.add(armGroup);

      if (isCentaur) {
        // Centaur Front Equine Legs with Shoulder Haunch & Hoof
        const equineHaunch = new THREE.Mesh(
          trackGeo(new THREE.SphereGeometry(0.18 * m.torsoBulk, seg(8, 5), seg(6, 4))),
          primaryArmorMat
        );
        equineHaunch.position.set(side * 0.26 * m.torsoBulk, pelvisY * 0.95, 0.42);
        exportableGroup.add(equineHaunch);

        const foreLeg = new THREE.Mesh(
          trackGeo(
            new THREE.CylinderGeometry(0.14 * m.torsoBulk, 0.1 * m.torsoBulk, legHeight, seg(8, 5))
          ),
          primaryArmorMat
        );
        foreLeg.position.set(side * 0.28 * m.torsoBulk, pelvisY * 0.5, 0.42);
        exportableGroup.add(foreLeg);
      }
    }

    const armLen = isQuadruped && !isCentaur ? legHeight : 0.78 * m.limbLength;

    // 1) Deltoid / Scapular Socket Sphere welding arm into torso/chest
    const deltoidGeo = trackGeo(
      new THREE.SphereGeometry(0.155 * m.torsoBulk, seg(8, 5), seg(6, 4))
    );
    deltoidGeo.scale(0.95, 1.15, 1.05);
    const deltoidMesh = new THREE.Mesh(deltoidGeo, primaryArmorMat);
    deltoidMesh.position.set(-side * 0.04, 0, 0);
    armGroup.add(deltoidMesh);

    // 2) Upper Arm / Bicep overlapping from shoulder down to elbow
    const upperArmGeo = trackGeo(
      new THREE.CylinderGeometry(0.135, 0.11, armLen * 0.52, seg(8, 5))
    );
    const upperArmMesh = new THREE.Mesh(
      upperArmGeo,
      isQuadruped ? primaryArmorMat : underbodyMat
    );
    upperArmMesh.position.y = -armLen * 0.24;
    armGroup.add(upperArmMesh);

    // 3) Articulated Elbow Sphere + Armored Couter Guard (Zero gap between bicep and forearm!)
    const elbowSphereGeo = trackGeo(new THREE.SphereGeometry(0.115, seg(7, 5), seg(5, 4)));
    const elbowSphere = new THREE.Mesh(elbowSphereGeo, secondaryTrimMat);
    elbowSphere.position.set(0, -armLen * 0.48, 0);
    armGroup.add(elbowSphere);

    // 4) Forearm / Vambrace overlapping from elbow down to wrist
    const forearmGeo = trackGeo(
      new THREE.CylinderGeometry(0.135, 0.105, armLen * 0.54, seg(8, 5))
    );
    const forearmMesh = new THREE.Mesh(forearmGeo, primaryArmorMat);
    forearmMesh.position.set(0, -armLen * 0.72, isQuadruped ? 0.02 : 0.05);
    forearmMesh.rotation.x = isQuadruped ? 0 : -0.22;
    armGroup.add(forearmMesh);

    // 5) Gauntlet / Clawed Front Paw
    const handGeo = trackGeo(new THREE.BoxGeometry(0.18, 0.16, 0.25));
    const handMesh = new THREE.Mesh(handGeo, secondaryTrimMat);
    handMesh.position.set(0, -armLen * 0.94, 0.1);
    armGroup.add(handMesh);

    const handSocket = new THREE.Object3D();
    handSocket.name = isLeft ? 'socket_shield_l' : 'socket_weapon_r';
    handSocket.position.set(0, -armLen * 0.94, 0.16);
    armGroup.add(handSocket);

    tagExplodablePart(armGroup, new THREE.Vector3(side * 0.65, 0.0, 0.25), {
      name: isQuadruped
        ? isLeft
          ? 'Extremidad Frontal Izquierda'
          : 'Extremidad Frontal Derecha'
        : isLeft
        ? 'Brazo Articulado Izquierdo'
        : 'Brazo Articulado Derecho',
      category: 'Secondary Read',
      vertexCount: 260,
      triangleCount: 210,
      materialSummary: 'Deltoides, bíceps, codera, brazal y guantelete continuos',
      socketName: handSocket.name,
    });
    explodableParts.push(armGroup);

    return { armGroup, handSocket };
  };

  const { armGroup: leftArmGroup, handSocket: socketShieldL } = buildArm(true);
  const { armGroup: rightArmGroup, handSocket: socketWeaponR } = buildArm(false);

  // 8. WEAPON / CLAWS ON socket_weapon_r
  const weaponGroup = new THREE.Group();
  weaponGroup.name = `Weapon_${config.weapon}`;
  weaponGroup.scale.setScalar(m.weaponScale);
  socketWeaponR.add(weaponGroup);

  if (config.weapon === 'feral_claws_fangs' || config.weapon === 'dragon_breath_maw') {
    for (let c = -1; c <= 1; c++) {
      const talonGeo = trackGeo(new THREE.ConeGeometry(0.036, 0.28, 5));
      talonGeo.rotateX(Math.PI / 2.2);
      const talonR = new THREE.Mesh(talonGeo, ivoryFangMat);
      talonR.position.set(c * 0.065, -0.02, 0.14);
      weaponGroup.add(talonR);

      const talonL = talonR.clone();
      socketShieldL.add(talonL);
    }
  } else {
    const shaftLen =
      config.weapon === 'arcane_staff' || config.weapon === 'radiant_halberd' ? 1.85 : 1.15;
    const shaftGeo = trackGeo(new THREE.CylinderGeometry(0.032, 0.038, shaftLen, seg(8, 5)));
    const shaftMesh = new THREE.Mesh(shaftGeo, underbodyMat);
    shaftMesh.position.set(0, 0.25, 0.25);
    shaftMesh.rotation.x = Math.PI / 2.4;
    weaponGroup.add(shaftMesh);

    // Pommel & Crossguard collar for weapon continuity
    const guardGeo = trackGeo(new THREE.BoxGeometry(0.28, 0.06, 0.09));
    const guardMesh = new THREE.Mesh(guardGeo, secondaryTrimMat);
    guardMesh.position.set(0, shaftLen * 0.45, 0);
    shaftMesh.add(guardMesh);

    if (config.weapon === 'runic_greatsword' || config.weapon === 'venom_scythe_claws') {
      const bladeShape = new THREE.Shape();
      bladeShape.moveTo(-0.085, 0);
      bladeShape.lineTo(0.085, 0);
      bladeShape.lineTo(0.055, 1.05);
      bladeShape.lineTo(0, 1.32);
      bladeShape.lineTo(-0.055, 1.05);
      bladeShape.closePath();

      const bladeGeo = trackGeo(
        new THREE.ExtrudeGeometry(bladeShape, {
          depth: 0.024,
          bevelEnabled: true,
          bevelThickness: 0.01,
          bevelSize: 0.012,
          bevelSegments: 1,
        })
      );
      bladeGeo.center();
      const bladeMesh = new THREE.Mesh(bladeGeo, bladeAlloyMat);
      bladeMesh.position.set(0, shaftLen * 0.5 + 0.52, 0);
      shaftMesh.add(bladeMesh);

      const runeStripGeo = trackGeo(new THREE.BoxGeometry(0.03, 0.85, 0.048));
      const runeStripMesh = new THREE.Mesh(runeStripGeo, emissiveRuneMat);
      runeStripMesh.position.set(0, shaftLen * 0.5 + 0.38, 0);
      shaftMesh.add(runeStripMesh);
    } else if (config.weapon === 'obsidian_warhammer') {
      const headGeo = trackGeo(new THREE.BoxGeometry(0.56, 0.34, 0.34));
      const hammerHead = new THREE.Mesh(headGeo, primaryArmorMat);
      hammerHead.position.set(0, shaftLen * 0.5 + 0.14, 0);
      shaftMesh.add(hammerHead);
    } else if (config.weapon === 'arcane_staff') {
      const finialGeo = trackGeo(new THREE.TorusGeometry(0.22, 0.035, 6, seg(16, 8)));
      const finialMesh = new THREE.Mesh(finialGeo, secondaryTrimMat);
      finialMesh.position.set(0, shaftLen * 0.5 + 0.2, 0);
      shaftMesh.add(finialMesh);

      const orbGeo = trackGeo(new THREE.OctahedronGeometry(0.12, density === 'low' ? 0 : 1));
      const orbMesh = new THREE.Mesh(orbGeo, crystalCoreMat);
      orbMesh.position.set(0, shaftLen * 0.5 + 0.2, 0);
      shaftMesh.add(orbMesh);
    } else {
      const axeShape = new THREE.Shape();
      axeShape.moveTo(0, -0.16);
      axeShape.quadraticCurveTo(0.42, -0.05, 0.48, 0.3);
      axeShape.quadraticCurveTo(0.25, 0.08, 0, 0.24);
      axeShape.closePath();

      const axeGeo = trackGeo(
        new THREE.ExtrudeGeometry(axeShape, {
          depth: 0.025,
          bevelEnabled: true,
          bevelThickness: 0.012,
          bevelSize: 0.012,
          bevelSegments: 1,
        })
      );
      const axeBladeR = new THREE.Mesh(axeGeo, bladeAlloyMat);
      axeBladeR.position.set(0.02, shaftLen * 0.45, -0.012);
      shaftMesh.add(axeBladeR);

      const axeBladeL = axeBladeR.clone();
      axeBladeL.scale.x = -1;
      axeBladeL.position.x = -0.02;
      shaftMesh.add(axeBladeL);
    }
  }

  tagExplodablePart(weaponGroup, new THREE.Vector3(-0.65, 0.15, 0.75), {
    name: `Armamento Principal (${config.weapon})`,
    category: 'Weapon & Socket',
    vertexCount: 310,
    triangleCount: 250,
    materialSummary: 'Anclado a socket_weapon_r con textura Damasco',
    socketName: 'socket_weapon_r',
  });
  explodableParts.push(weaponGroup);

  // 9. OFF-HAND SHIELD / ORB ON socket_shield_l
  const offhandGroup = new THREE.Group();
  offhandGroup.name = `Offhand_${config.offhand}`;
  socketShieldL.add(offhandGroup);

  if (config.offhand === 'tower_shield' || config.offhand === 'runic_buckler') {
    const isTower = config.offhand === 'tower_shield';
    const shW = isTower ? 0.42 : 0.32;
    const shH = isTower ? 0.88 : 0.42;
    const shieldShape = new THREE.Shape();
    shieldShape.moveTo(-shW, shH * 0.5);
    shieldShape.lineTo(shW, shH * 0.5);
    shieldShape.lineTo(shW * 0.85, -shH * 0.25);
    shieldShape.lineTo(0, -shH * 0.6);
    shieldShape.lineTo(-shW * 0.85, -shH * 0.25);
    shieldShape.closePath();

    const shieldGeo = trackGeo(
      new THREE.ExtrudeGeometry(shieldShape, {
        depth: 0.06,
        bevelEnabled: true,
        bevelThickness: 0.02,
        bevelSize: 0.02,
        bevelSegments: 1,
      })
    );
    const shieldMesh = new THREE.Mesh(shieldGeo, primaryArmorMat);
    shieldMesh.position.set(0.16, 0, 0.12);
    shieldMesh.rotation.y = 0.45;
    offhandGroup.add(shieldMesh);
  } else if (config.offhand === 'phylactery_orb') {
    const censerGeo = trackGeo(
      new THREE.IcosahedronGeometry(0.16, density === 'low' ? 0 : 1)
    );
    const censerMesh = new THREE.Mesh(censerGeo, crystalCoreMat);
    censerMesh.position.set(0.1, 0.08, 0.24);
    offhandGroup.add(censerMesh);
  } else if (config.offhand === 'dual_blade') {
    const daggerGeo = trackGeo(new THREE.ConeGeometry(0.08, 0.85, seg(6, 4)));
    const daggerMesh = new THREE.Mesh(daggerGeo, bladeAlloyMat);
    daggerMesh.position.set(0, 0.1, 0.42);
    daggerMesh.rotation.x = Math.PI / 2;
    offhandGroup.add(daggerMesh);
  }

  // 10. ARTICULATED WINGS & DORSAL STRUCTURES WITH SCAPULAR ROOT SOCKETS
  const dorsalGroup = new THREE.Group();
  dorsalGroup.name = `Dorsal_${config.dorsal}`;
  dorsalGroup.position.set(0, torsoHeight * 0.68, -0.18 * m.torsoBulk);
  dorsalGroup.scale.setScalar(m.dorsalScale);
  torsoGroup.add(dorsalGroup);

  const leftWingGroup = new THREE.Group();
  leftWingGroup.name = 'Wing_L';
  const rightWingGroup = new THREE.Group();
  rightWingGroup.name = 'Wing_R';
  dorsalGroup.add(leftWingGroup);
  dorsalGroup.add(rightWingGroup);

  if (config.dorsal === 'dragon_membrane_wings') {
    for (const side of [-1, 1]) {
      const targetWing = side === 1 ? leftWingGroup : rightWingGroup;

      // Scapular Wing Root Muscle welding wing into the dorsal spine
      const wingRootGeo = trackGeo(new THREE.SphereGeometry(0.13, seg(6, 4), seg(5, 4)));
      const wingRoot = new THREE.Mesh(wingRootGeo, secondaryTrimMat);
      wingRoot.position.set(side * 0.08, 0.02, 0.04);
      targetWing.add(wingRoot);

      const sparCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(side * 0.04, 0, 0.04),
        new THREE.Vector3(side * 0.55, 0.48, -0.08),
        new THREE.Vector3(side * 1.25, 0.62, -0.16),
        new THREE.Vector3(side * 1.65, 0.15, -0.22),
      ]);
      const sparGeo = trackGeo(
        new THREE.TubeGeometry(sparCurve, seg(10, 6), 0.044, seg(6, 4), false)
      );
      targetWing.add(new THREE.Mesh(sparGeo, primaryArmorMat));

      const webShape = new THREE.Shape();
      webShape.moveTo(0, 0);
      webShape.lineTo(side * 0.55, 0.46);
      webShape.lineTo(side * 1.25, 0.6);
      webShape.lineTo(side * 1.62, 0.14);
      webShape.quadraticCurveTo(side * 1.15, -0.15, side * 0.85, -0.45);
      webShape.quadraticCurveTo(side * 0.45, -0.2, 0, -0.35);
      webShape.closePath();

      const webGeo = trackGeo(
        new THREE.ExtrudeGeometry(webShape, { depth: 0.02, bevelEnabled: false })
      );
      const webMesh = new THREE.Mesh(webGeo, secondaryTrimMat);
      webMesh.position.z = -0.08;
      targetWing.add(webMesh);
    }
  } else if (config.dorsal === 'seraph_wings') {
    for (const side of [-1, 1]) {
      const targetWing = side === 1 ? leftWingGroup : rightWingGroup;
      const wingArchGeo = trackGeo(new THREE.SphereGeometry(0.12, 6, 5));
      wingArchGeo.scale(1.8, 0.8, 0.8);
      const wingArch = new THREE.Mesh(wingArchGeo, secondaryTrimMat);
      wingArch.position.set(side * 0.15, 0.15, 0);
      targetWing.add(wingArch);

      for (let f = 0; f < 6; f++) {
        const featherGeo = trackGeo(new THREE.BoxGeometry(0.09, 0.85 - f * 0.1, 0.03));
        const featherMesh = new THREE.Mesh(
          featherGeo,
          f % 2 === 0 ? secondaryTrimMat : emissiveRuneMat
        );
        featherMesh.position.set(side * (0.22 + f * 0.16), 0.22 - f * 0.05, -0.03 * f);
        featherMesh.rotation.z = -side * (0.42 + f * 0.15);
        targetWing.add(featherMesh);
      }
    }
  } else if (config.dorsal === 'beholder_eye_tentacles') {
    for (let i = 0; i < 8; i++) {
      const ang = (i / 8) * Math.PI * 2;
      const rx = Math.cos(ang);
      const rz = Math.sin(ang);
      const stalkCurve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(rx * 0.18, 0.05, rz * 0.18),
        new THREE.Vector3(rx * 0.65, 0.45 + (i % 2) * 0.18, rz * 0.65),
        new THREE.Vector3(rx * 0.88, 0.28 + (i % 3) * 0.15, rz * 0.88),
      ]);
      const stalkGeo = trackGeo(
        new THREE.TubeGeometry(stalkCurve, seg(8, 5), 0.04, seg(5, 4), false)
      );
      dorsalGroup.add(new THREE.Mesh(stalkGeo, primaryArmorMat));

      const tipEye = new THREE.Mesh(
        trackGeo(new THREE.SphereGeometry(0.068, seg(8, 5), seg(6, 4))),
        emissiveRuneMat
      );
      tipEye.position.set(rx * 0.88, 0.28 + (i % 3) * 0.15, rz * 0.88);
      dorsalGroup.add(tipEye);
    }
  } else if (config.dorsal === 'tattered_mantle') {
    const capeShape = new THREE.Shape();
    capeShape.moveTo(-0.42, 0.24);
    capeShape.lineTo(0.42, 0.24);
    capeShape.lineTo(0.58, -1.25);
    capeShape.lineTo(0, -1.32);
    capeShape.lineTo(-0.58, -1.25);
    capeShape.closePath();
    const capeGeo = trackGeo(
      new THREE.ExtrudeGeometry(capeShape, { depth: 0.03, bevelEnabled: false })
    );
    const capeMesh = new THREE.Mesh(capeGeo, underbodyMat);
    capeMesh.rotation.x = 0.14;
    dorsalGroup.add(capeMesh);
  } else if (config.dorsal === 'crystal_pylons' || config.dorsal === 'bio_chitin_spines') {
    for (let i = -2; i <= 2; i++) {
      if (i === 0) continue;
      const pylonGeo = trackGeo(new THREE.OctahedronGeometry(0.11, 0));
      pylonGeo.scale(0.8, 3.4, 0.8);
      const pylonMesh = new THREE.Mesh(
        pylonGeo,
        config.dorsal === 'crystal_pylons' ? crystalCoreMat : secondaryTrimMat
      );
      pylonMesh.position.set(i * 0.16, 0.22 + (2 - Math.abs(i)) * 0.12, -0.04);
      pylonMesh.rotation.z = -i * 0.25;
      pylonMesh.rotation.x = -0.22;
      dorsalGroup.add(pylonMesh);
    }
  } else if (config.dorsal === 'runic_halo_ring') {
    const haloGeo = trackGeo(new THREE.TorusGeometry(0.52, 0.035, 6, seg(20, 12)));
    const haloMesh = new THREE.Mesh(haloGeo, emissiveRuneMat);
    haloMesh.position.set(0, 0.28, -0.04);
    dorsalGroup.add(haloMesh);
  }

  tagExplodablePart(dorsalGroup, new THREE.Vector3(0, 0.55, -0.85), {
    name: `Alas / Estructura Dorsal (${config.dorsal})`,
    category: 'Secondary Read',
    vertexCount: 380,
    triangleCount: 310,
    materialSummary: 'Unión escapular continua al torso dorsal',
  });
  explodableParts.push(dorsalGroup);

  // 11. CONTINUOUS ARTICULATED TAIL WITH CAUDAL BASE COLLAR
  const tailGroup = new THREE.Group();
  tailGroup.name = 'Articulated_Tail';
  tailGroup.position.set(0, 0.06, isQuadruped || isArachnid ? -0.52 : -0.22 * m.torsoBulk);
  pelvisGroup.add(tailGroup);

  if (m.tailLength > 0.08 && !isSerpentine) {
    const tLen = 1.15 * m.tailLength;
    const isScorpion =
      config.dorsal === 'scorpion_stinger_tail' ||
      isArachnid ||
      config.archetype === 'manticore_chimera';

    // Caudal Root Musculature Sphere welding tail into pelvis/abdomen
    const tailRootGeo = trackGeo(
      new THREE.SphereGeometry(0.15 * m.torsoBulk, seg(8, 5), seg(6, 4))
    );
    tailRootGeo.scale(1.0, 0.9, 1.35);
    const tailRootMesh = new THREE.Mesh(tailRootGeo, primaryArmorMat);
    tailRootMesh.position.set(0, 0, 0.06);
    tailGroup.add(tailRootMesh);

    const tailCurve = isScorpion
      ? new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0.08),
          new THREE.Vector3(0, 0.45 * tLen, -0.45 * tLen),
          new THREE.Vector3(0, 1.05 * tLen, -0.25 * tLen),
          new THREE.Vector3(0, 1.25 * tLen, 0.25 * tLen),
        ])
      : new THREE.CatmullRomCurve3([
          new THREE.Vector3(0, 0, 0.08),
          new THREE.Vector3(0, -0.16, -0.45 * tLen),
          new THREE.Vector3(0, -0.22, -0.95 * tLen),
          new THREE.Vector3(0, -0.1, -1.35 * tLen),
        ]);

    const isBushyTail =
      config.archetype === 'furious_keltir' ||
      config.archetype === 'rabid_grey_wolf' ||
      config.archetype === 'dire_wolf';

    const tailGeo = trackGeo(
      new THREE.TubeGeometry(
        tailCurve,
        seg(12, 7),
        (isBushyTail ? 0.165 : 0.11) * m.torsoBulk,
        seg(7, 4),
        false
      )
    );
    const tailMesh = new THREE.Mesh(tailGeo, primaryArmorMat);
    tailGroup.add(tailMesh);

    const tipPt = tailCurve.getPoint(1);
    const stingerGeo = trackGeo(
      new THREE.ConeGeometry(isBushyTail ? 0.18 : 0.13, isBushyTail ? 0.44 : 0.36, 6)
    );
    stingerGeo.rotateX(isScorpion ? Math.PI * 0.65 : -Math.PI / 2);
    const stingerMesh = new THREE.Mesh(
      stingerGeo,
      isScorpion ? emissiveRuneMat : secondaryTrimMat
    );
    stingerMesh.position.copy(tipPt);
    tailGroup.add(stingerMesh);

    tagExplodablePart(tailGroup, new THREE.Vector3(0, -0.1, -0.95), {
      name: isScorpion ? 'Aguijón Caudal Articulado' : 'Cola Dracónica / Bestial Continua',
      category: 'Secondary Read',
      vertexCount: 220,
      triangleCount: 180,
      materialSummary: 'Base caudal fusionada a la pelvis + aguijón/maza',
    });
    explodableParts.push(tailGroup);
  }

  // 12. INSTANCED ORBITING RUNIC SHARDS
  let auraInstancedMesh: THREE.InstancedMesh | null = null;
  const shardData: BuiltEnemyRig['shardData'] = [];
  const rawShardCount = Math.max(0, Math.round(m.auraShardCount));
  const shardCount =
    density === 'low' ? Math.min(8, Math.ceil(rawShardCount * 0.5)) : rawShardCount;

  if (shardCount > 0) {
    const shardGeo = trackGeo(new THREE.OctahedronGeometry(0.055, 0));
    shardGeo.scale(0.7, 1.8, 0.7);
    auraInstancedMesh = new THREE.InstancedMesh(shardGeo, emissiveRuneMat, shardCount);
    auraInstancedMesh.name = 'Aura_Runic_Shards';
    auraInstancedMesh.frustumCulled = false;

    const baseColor = new THREE.Color(matCfg.emissiveColor);
    for (let i = 0; i < shardCount; i++) {
      const angle = (i / shardCount) * Math.PI * 2;
      const radius = 0.85 + (i % 3) * 0.25;
      const yOffset = 0.35 + ((i * 7) % 10) * 0.18;
      const speed = 0.6 + (i % 5) * 0.18;
      const scale = 0.85 + ((i * 13) % 30) / 100;

      shardData.push({ angle, radius, yOffset, speed, scale });

      _dummyObj.position.set(Math.cos(angle) * radius, yOffset, Math.sin(angle) * radius);
      _dummyObj.rotation.set(0.2, angle, 0.15);
      _dummyObj.scale.setScalar(scale);
      _dummyObj.updateMatrix();
      auraInstancedMesh.setMatrixAt(i, _dummyObj.matrix);

      _shardColor.copy(baseColor).offsetHSL(((i % 5) - 2) * 0.03, 0, ((i % 3) - 1) * 0.08);
      auraInstancedMesh.setColorAt(i, _shardColor);
    }
    auraInstancedMesh.instanceMatrix.needsUpdate = true;
    if (auraInstancedMesh.instanceColor) auraInstancedMesh.instanceColor.needsUpdate = true;
    auraInstancedMesh.computeBoundingSphere();
    exportableGroup.add(auraInstancedMesh);
  }

  // 13. HITBOXES, SOCKETS & AGGRO RANGE VISUALIZER GROUP
  const hitboxGroup = new THREE.Group();
  hitboxGroup.name = 'Tactical_Hitboxes_And_Sockets';
  rootGroup.add(hitboxGroup);

  const hitboxWireMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: 0x10b981,
      wireframe: true,
      transparent: true,
      opacity: 0.45,
      depthWrite: false,
    })
  );
  const headHitboxMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: 0xf43f5e,
      wireframe: true,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
    })
  );

  const torsoColGeo = trackGeo(
    new THREE.CylinderGeometry(
      config.stats.torsoHitboxRadius,
      config.stats.torsoHitboxRadius,
      config.stats.torsoHitboxHeight,
      12,
      1,
      true
    )
  );
  const torsoColMesh = new THREE.Mesh(torsoColGeo, hitboxWireMat);
  torsoColMesh.position.set(0, config.stats.torsoHitboxHeight * 0.5 + 0.25, 0);
  hitboxGroup.add(torsoColMesh);

  const headColGeo = trackGeo(new THREE.SphereGeometry(config.stats.headHitboxRadius, 10, 8));
  const headColMesh = new THREE.Mesh(headColGeo, headHitboxMat);
  const headWorldPos = new THREE.Vector3();
  socketHead.getWorldPosition(headWorldPos);
  headColMesh.position.copy(headWorldPos);
  hitboxGroup.add(headColMesh);

  const rangeRingGeo = trackGeo(
    new THREE.RingGeometry(
      Math.max(0.5, config.stats.attackRange * 0.35 - 0.04),
      config.stats.attackRange * 0.35,
      36
    )
  );
  rangeRingGeo.rotateX(-Math.PI / 2);
  const rangeRingMat = trackMat(
    new THREE.MeshBasicMaterial({
      color: 0xf59e0b,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0.65,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    })
  );
  const rangeRingMesh = new THREE.Mesh(rangeRingGeo, rangeRingMat);
  rangeRingMesh.position.y = 0.03;
  hitboxGroup.add(rangeRingMesh);

  const addSocketGizmo = (socket: THREE.Object3D) => {
    const axes = new THREE.AxesHelper(0.28);
    socket.add(axes);
  };
  addSocketGizmo(socketHead);
  addSocketGizmo(socketWeaponR);
  addSocketGizmo(socketShieldL);
  addSocketGizmo(socketCore);

  // Compute exact vertex and triangle counts
  let totalVertices = 0;
  let totalTriangles = 0;
  let partCount = 0;
  exportableGroup.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    if (mesh.isMesh && mesh.geometry) {
      partCount++;
      const posAttr = mesh.geometry.getAttribute('position');
      if (posAttr) {
        const instanceMult = (mesh as THREE.InstancedMesh).isInstancedMesh
          ? (mesh as THREE.InstancedMesh).count
          : 1;
        totalVertices += posAttr.count * instanceMult;
        const index = mesh.geometry.getIndex();
        if (index) {
          totalTriangles += (index.count / 3) * instanceMult;
        } else {
          totalTriangles += (posAttr.count / 3) * instanceMult;
        }
      }
    }
  });

  return {
    rootGroup,
    exportableGroup,
    hitboxGroup,
    materials: {
      primaryArmorMat,
      secondaryTrimMat,
      underbodyMat,
      emissiveRuneMat,
      crystalCoreMat,
    },
    parts: {
      pelvisGroup,
      torsoGroup,
      neckGroup,
      headGroup,
      jawGroup,
      extraHeads,
      leftShoulderGroup,
      rightShoulderGroup,
      leftArmGroup,
      rightArmGroup,
      leftLegGroup,
      rightLegGroup,
      leftWingGroup,
      rightWingGroup,
      tailGroup,
      dorsalGroup,
      weaponGroup,
      offhandGroup,
      coreMesh,
      breathConeMesh,
      auraInstancedMesh,
    },
    sockets: {
      head: socketHead,
      weaponR: socketWeaponR,
      shieldL: socketShieldL,
      core: socketCore,
    },
    explodableParts,
    shardData,
    metrics: {
      totalVertices: Math.round(totalVertices),
      totalTriangles: Math.round(totalTriangles),
      partCount,
    },
    dispose: () => {
      geometriesToDispose.forEach((g) => g.dispose());
      materialsToDispose.forEach((mItem) => mItem.dispose());
    },
  };
}

export function updateEnemyAnimation(
  rig: BuiltEnemyRig,
  animState: AnimationState,
  elapsed: number,
  delta: number,
  speedMultiplier: number,
  explodedAmount: number
) {
  const t = elapsed * speedMultiplier;
  const {
    pelvisGroup,
    neckGroup,
    headGroup,
    jawGroup,
    extraHeads,
    leftShoulderGroup,
    rightShoulderGroup,
    leftArmGroup,
    rightArmGroup,
    leftLegGroup,
    rightLegGroup,
    leftWingGroup,
    rightWingGroup,
    tailGroup,
    dorsalGroup,
    coreMesh,
    breathConeMesh,
    auraInstancedMesh,
  } = rig.parts;

  const bodyPlan = rig.exportableGroup.userData.bodyPlan as string;
  const isFloating = bodyPlan === 'floating_magical';
  const isDragon = bodyPlan === 'dragon_wyvern';

  // 1. Smoothly interpolate Exploded View separations
  for (let i = 0; i < rig.explodableParts.length; i++) {
    const part = rig.explodableParts[i];
    const basePos = part.userData.basePos as THREE.Vector3;
    const explodeVec = part.userData.explodeVec as THREE.Vector3;
    if (basePos && explodeVec) {
      _targetPos.copy(basePos).addScaledVector(explodeVec, explodedAmount);
      part.position.x = THREE.MathUtils.damp(part.position.x, _targetPos.x, 8, delta);
      part.position.y = THREE.MathUtils.damp(part.position.y, _targetPos.y, 8, delta);
      part.position.z = THREE.MathUtils.damp(part.position.z, _targetPos.z, 8, delta);
    }
  }

  if (isFloating && pelvisGroup.userData.basePos) {
    pelvisGroup.position.y = pelvisGroup.userData.basePos.y + Math.sin(t * 2.4) * 0.14;
  }

  // 2. Rotate Runic Core Crystal & Update Instanced Aura Shards
  coreMesh.rotation.y = t * 1.4;
  coreMesh.rotation.z = Math.sin(t * 2.0) * 0.25;

  if (auraInstancedMesh && rig.shardData.length > 0) {
    for (let i = 0; i < rig.shardData.length; i++) {
      const s = rig.shardData[i];
      const curAngle = s.angle + t * s.speed;
      const r = s.radius + explodedAmount * 0.45;
      const y = s.yOffset + Math.sin(t * 2.2 + i) * 0.12;
      _dummyObj.position.set(Math.cos(curAngle) * r, y, Math.sin(curAngle) * r);
      _dummyObj.rotation.set(t * 0.8 + i, curAngle, 0.25);
      _dummyObj.scale.setScalar(s.scale);
      _dummyObj.updateMatrix();
      auraInstancedMesh.setMatrixAt(i, _dummyObj.matrix);
    }
    auraInstancedMesh.instanceMatrix.needsUpdate = true;
  }

  // Continuous Tail, Extra Heads & Wing Secondary Motion
  const tailWave = Math.sin(t * 3.0);
  tailGroup.rotation.y = THREE.MathUtils.damp(tailGroup.rotation.y, tailWave * 0.26, 8, delta);

  for (let i = 0; i < extraHeads.length; i++) {
    const eh = extraHeads[i];
    const phase = i === 0 ? 1.4 : -1.4;
    eh.rotation.x = Math.sin(t * 2.4 + phase) * 0.12;
  }

  const wingFlap = Math.sin(t * (animState === 'attack' ? 6.5 : 3.2));
  leftWingGroup.rotation.y = THREE.MathUtils.damp(leftWingGroup.rotation.y, wingFlap * 0.42, 10, delta);
  rightWingGroup.rotation.y = THREE.MathUtils.damp(rightWingGroup.rotation.y, -wingFlap * 0.42, 10, delta);

  const dampLambda = 10;

  if (animState === 'idle') {
    const breath = Math.sin(t * 2.2);
    breathConeMesh.visible = false;
    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, Math.max(0, breath * 0.12), dampLambda, delta);
    neckGroup.rotation.y = THREE.MathUtils.damp(neckGroup.rotation.y, Math.sin(t * 1.1) * 0.14, dampLambda, delta);
    headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, Math.sin(t * 1.1) * 0.12, dampLambda, delta);
    headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -breath * 0.04, dampLambda, delta);

    leftShoulderGroup.rotation.z = THREE.MathUtils.damp(leftShoulderGroup.rotation.z, breath * 0.05, dampLambda, delta);
    rightShoulderGroup.rotation.z = THREE.MathUtils.damp(rightShoulderGroup.rotation.z, -breath * 0.05, dampLambda, delta);

    rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -0.15 + breath * 0.06, dampLambda, delta);
    leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.18 - breath * 0.05, dampLambda, delta);

    leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, 0, dampLambda, delta);
    rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, 0, dampLambda, delta);
    pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 0, dampLambda, delta);
    tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, 0, dampLambda, delta);
  } else if (animState === 'walk') {
    breathConeMesh.visible = false;
    const stride = Math.sin(t * 5.6);
    const bounce = Math.abs(Math.cos(t * 5.6)) * 0.06;

    pelvisGroup.position.y += bounce;
    headGroup.rotation.y = THREE.MathUtils.damp(headGroup.rotation.y, stride * 0.1, dampLambda, delta);
    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.15, dampLambda, delta);

    leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, stride * 0.55, dampLambda, delta);
    rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, -stride * 0.55, dampLambda, delta);

    rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, stride * 0.48 - 0.15, dampLambda, delta);
    leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -stride * 0.48 - 0.15, dampLambda, delta);
    dorsalGroup.rotation.x = THREE.MathUtils.damp(dorsalGroup.rotation.x, 0.14 + Math.abs(stride) * 0.1, dampLambda, delta);
  } else if (animState === 'attack') {
    const cycle = (t * 2.5) % (Math.PI * 2);
    const swing = Math.sin(cycle);
    const lunge = Math.max(0, Math.sin(cycle - 0.4));

    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, lunge * 0.65, 16, delta);
    if (isDragon || isFloating || bodyPlan === 'serpentine_naga') {
      breathConeMesh.visible = lunge > 0.15;
      const bScale = Math.max(0.01, lunge * 1.15);
      breathConeMesh.scale.set(bScale, bScale, bScale * 1.3);
      breathConeMesh.rotation.z = t * 6.0;
    } else {
      breathConeMesh.visible = false;
    }

    tailGroup.rotation.x = THREE.MathUtils.damp(tailGroup.rotation.x, lunge * 0.65, 14, delta);

    rightArmGroup.rotation.x = THREE.MathUtils.damp(
      rightArmGroup.rotation.x,
      swing > 0 ? -1.55 + lunge * 2.35 : -0.35,
      16,
      delta
    );
    leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -0.65 + swing * 0.35, 12, delta);
    leftLegGroup.rotation.x = THREE.MathUtils.damp(leftLegGroup.rotation.x, -lunge * 0.32, 12, delta);
    rightLegGroup.rotation.x = THREE.MathUtils.damp(rightLegGroup.rotation.x, lunge * 0.32, 12, delta);
  } else if (animState === 'hit') {
    breathConeMesh.visible = false;
    const stagger = Math.sin(t * 9.0) * 0.25;
    jawGroup.rotation.x = THREE.MathUtils.damp(jawGroup.rotation.x, 0.45, 16, delta);
    headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, -0.38 + stagger, 16, delta);
    rightArmGroup.rotation.z = THREE.MathUtils.damp(rightArmGroup.rotation.z, -0.5, 14, delta);
    leftArmGroup.rotation.z = THREE.MathUtils.damp(leftArmGroup.rotation.z, 0.5, 14, delta);
  } else if (animState === 'death') {
    breathConeMesh.visible = false;
    pelvisGroup.rotation.x = THREE.MathUtils.damp(pelvisGroup.rotation.x, 1.05, 5, delta);
    headGroup.rotation.x = THREE.MathUtils.damp(headGroup.rotation.x, 0.45, 5, delta);
    rightArmGroup.rotation.x = THREE.MathUtils.damp(rightArmGroup.rotation.x, -1.15, 5, delta);
    leftArmGroup.rotation.x = THREE.MathUtils.damp(leftArmGroup.rotation.x, -1.05, 5, delta);
  }
}
