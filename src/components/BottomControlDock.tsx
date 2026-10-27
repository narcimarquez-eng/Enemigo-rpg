import React, { useState } from 'react';
import {
  Box,
  Camera,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ChevronUp,
  Code2,
  Download,
  Eye,
  Grid,
  Layers,
  Maximize2,
  Minimize2,
  Palette,
  Play,
  Shield,
  Shuffle,
  Sliders,
  Sparkles,
  Sun,
  Swords,
} from 'lucide-react';
import {
  ARCHETYPE_PRESETS,
  AiBehaviorType,
  AnimationState,
  ArchetypeId,
  BodyPlan,
  CameraPresetId,
  DorsalType,
  ElementAffinity,
  EnemyConfig,
  EnemyTier,
  HelmStyle,
  LightingPresetId,
  MATERIAL_PRESETS,
  MaterialPresetId,
  MeshDensity,
  OffhandType,
  PartInspectionInfo,
  RACE_CATEGORY_LABELS,
  RaceCategory,
  TexturePatternId,
  WeaponType,
  calculateThreatScore,
  getEffectiveMeshDensity,
  getEffectiveTexturePattern,
} from '../types/enemy';

export type DockTabId = 'catalog' | 'anatomy' | 'equipment' | 'scene' | 'rpg';

interface BottomControlDockProps {
  config: EnemyConfig;
  onChangeConfig: (updater: (prev: EnemyConfig) => EnemyConfig) => void;
  activeTab: DockTabId;
  onTabChange: (tab: DockTabId) => void;
  isCollapsed: boolean;
  onToggleCollapse: () => void;
  explodedAmount: number;
  onExplodedChange: (val: number) => void;
  animState: AnimationState;
  onAnimStateChange: (state: AnimationState) => void;
  animSpeed: number;
  onAnimSpeedChange: (speed: number) => void;
  cameraPreset: CameraPresetId;
  onCameraPresetChange: (preset: CameraPresetId) => void;
  lightingPreset: LightingPresetId;
  onLightingPresetChange: (preset: LightingPresetId) => void;
  showHitboxes: boolean;
  onToggleHitboxes: () => void;
  wireframe: boolean;
  onToggleWireframe: () => void;
  autoRotate: boolean;
  onToggleAutoRotate: () => void;
  onRandomize: () => void;
  onQuickExportGLB: () => void;
  onOpenExportModal: () => void;
  inspectedPart: PartInspectionInfo | null;
}

const ALL_ARCHETYPE_KEYS = Object.keys(ARCHETYPE_PRESETS) as ArchetypeId[];

const BODY_PLAN_OPTIONS: Array<{ id: BodyPlan; label: string }> = [
  { id: 'dragon_wyvern', label: 'Dragón / Wyvern Alado' },
  { id: 'quadruped_beast', label: 'Bestia Cuadrúpeda / Centauro' },
  { id: 'serpentine_naga', label: 'Serpentino / Hidra / Gorgona' },
  { id: 'arachnid_scorpion', label: 'Arácnido / Escorpión (8 Patas)' },
  { id: 'floating_magical', label: 'Entidad Mágica Levitante' },
  { id: 'biped_titan', label: 'Titán Bípedo / Minotauro' },
];

const WEAPON_OPTIONS: Array<{ id: WeaponType; label: string }> = [
  { id: 'dragon_breath_maw', label: 'Aliento Elemental y Fauces' },
  { id: 'feral_claws_fangs', label: 'Zarpas y Colmillos Salvajes' },
  { id: 'runic_greatsword', label: 'Mandoble Rúnico' },
  { id: 'obsidian_warhammer', label: 'Martillo de Asedio' },
  { id: 'arcane_staff', label: 'Bastón Arcano' },
  { id: 'twin_reaver_axes', label: 'Labrys / Hacha Doble' },
  { id: 'radiant_halberd', label: 'Alabarda Celestial' },
  { id: 'venom_scythe_claws', label: 'Guadaña de Quitina' },
];

const OFFHAND_OPTIONS: Array<{ id: OffhandType; label: string }> = [
  { id: 'none', label: 'Sin Escudo / Zarpa Libre' },
  { id: 'tower_shield', label: 'Escudo de Torre' },
  { id: 'runic_buckler', label: 'Rodela Rúnica' },
  { id: 'phylactery_orb', label: 'Orbe Filacteria' },
  { id: 'dual_blade', label: 'Hoja / Garra Secundaria' },
];

const HELM_OPTIONS: Array<{ id: HelmStyle; label: string }> = [
  { id: 'draconic_snout', label: 'Cráneo y Fauces de Dragón' },
  { id: 'lupine_maw', label: 'Morro de Lobo / Bestia' },
  { id: 'tauren_bull_horns', label: 'Testuz de Minotauro / Demonio' },
  { id: 'avian_phoenix_beak', label: 'Pico Real de Fénix / Grifo' },
  { id: 'beholder_cyclops_eye', label: 'Ojo Cíclope de Tirano' },
  { id: 'horned_greathelm', label: 'Gran Yelmo Cornudo / Oni' },
  { id: 'crowned_skull', label: 'Cráneo Coronado / Liche' },
  { id: 'monolith_visor', label: 'Monolito de Gólem' },
  { id: 'valkyrie_crest', label: 'Cresta de Valquiria / Djinn' },
  { id: 'arachnid_mandibles', label: 'Quelíceros Arácnidos' },
  { id: 'drake_fang_helm', label: 'Yelmo Sangredragón' },
];

const DORSAL_OPTIONS: Array<{ id: DorsalType; label: string }> = [
  { id: 'dragon_membrane_wings', label: 'Alas Membranosas de Dragón' },
  { id: 'seraph_wings', label: 'Alas de Plumas Solares / Grifo' },
  { id: 'scorpion_stinger_tail', label: 'Aguijón Caudal de Escorpión' },
  { id: 'beholder_eye_tentacles', label: 'Tentáculos Oculares / Sierpes' },
  { id: 'crystal_pylons', label: 'Pilones de Cristal Dorsal' },
  { id: 'bio_chitin_spines', label: 'Cresta de Espinas Bestiales' },
  { id: 'tattered_mantle', label: 'Manto de Guerra' },
  { id: 'runic_halo_ring', label: 'Halo Rúnico' },
  { id: 'none', label: 'Sin Estructura Dorsal' },
];

export const BottomControlDock: React.FC<BottomControlDockProps> = ({
  config,
  onChangeConfig,
  activeTab,
  onTabChange,
  isCollapsed,
  onToggleCollapse,
  explodedAmount,
  onExplodedChange,
  animState,
  onAnimStateChange,
  animSpeed,
  onAnimSpeedChange,
  cameraPreset,
  onCameraPresetChange,
  lightingPreset,
  onLightingPresetChange,
  showHitboxes,
  onToggleHitboxes,
  wireframe,
  onToggleWireframe,
  autoRotate,
  onToggleAutoRotate,
  onRandomize,
  onQuickExportGLB,
  onOpenExportModal,
  inspectedPart,
}) => {
  const [selectedCategory, setSelectedCategory] = useState<RaceCategory | 'all'>('all');
  const [isTallDock, setIsTallDock] = useState(false);

  const threatScore = calculateThreatScore(config.stats);
  const currentIndex = ALL_ARCHETYPE_KEYS.indexOf(config.archetype);

  const handleStepArchetype = (dir: -1 | 1) => {
    const nextIdx =
      (currentIndex + dir + ALL_ARCHETYPE_KEYS.length) % ALL_ARCHETYPE_KEYS.length;
    const nextKey = ALL_ARCHETYPE_KEYS[nextIdx];
    onChangeConfig(() => structuredClone(ARCHETYPE_PRESETS[nextKey]));
  };

  const filteredArchetypes = ALL_ARCHETYPE_KEYS.filter(
    (archId) =>
      selectedCategory === 'all' || ARCHETYPE_PRESETS[archId].raceCategory === selectedCategory
  );

  return (
    <div className="fixed inset-x-0 bottom-3 z-30 mx-auto w-[min(1080px,95vw)] pointer-events-auto">
      {/* Top Bar of Dock: Quick Character Selector + Stepper + Collapse / Expand Buttons */}
      <div className="flex flex-wrap items-center justify-between gap-2 mb-2 px-1">
        {/* Left: Prev/Next Character Stepper + Direct Select Dropdown of all 30 Characters */}
        <div className="flex flex-wrap items-center gap-2 bg-[#111622] border border-slate-600/90 rounded-lg px-2.5 py-1.5 shadow-xl">
          <button
            onClick={() => handleStepArchetype(-1)}
            className="p-1.5 text-slate-200 hover:text-amber-400 hover:bg-slate-800 rounded transition-colors cursor-pointer"
            title="Personaje anterior"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>

          <select
            value={config.archetype}
            onChange={(e) => {
              const chosen = e.target.value as ArchetypeId;
              onChangeConfig(() => structuredClone(ARCHETYPE_PRESETS[chosen]));
            }}
            aria-label="Seleccionar personaje"
            className="bg-[#1b2336] border border-slate-600 hover:border-amber-400 rounded-md px-2.5 py-1 text-xs sm:text-sm font-bold text-amber-400 focus:outline-none focus:border-amber-400 cursor-pointer"
          >
            {(Object.keys(RACE_CATEGORY_LABELS) as RaceCategory[]).map((cat) => (
              <optgroup key={cat} label={RACE_CATEGORY_LABELS[cat]}>
                {ALL_ARCHETYPE_KEYS.filter(
                  (k) => ARCHETYPE_PRESETS[k].raceCategory === cat
                ).map((archKey) => {
                  const p = ARCHETYPE_PRESETS[archKey];
                  const idx = ALL_ARCHETYPE_KEYS.indexOf(archKey) + 1;
                  return (
                    <option key={archKey} value={archKey}>
                      {idx}. {p.name} — {p.title}
                    </option>
                  );
                })}
              </optgroup>
            ))}
          </select>

          <button
            onClick={() => handleStepArchetype(1)}
            className="p-1.5 text-slate-200 hover:text-amber-400 hover:bg-slate-800 rounded transition-colors cursor-pointer"
            title="Siguiente personaje"
          >
            <ChevronRight className="w-4 h-4" />
          </button>

          <span aria-hidden="true" className="text-slate-600 hidden md:inline">
            |
          </span>
          <span className="text-xs text-slate-200 hidden md:inline">
            {config.stats.tier} Nv.{config.stats.level}
          </span>
          {inspectedPart && (
            <>
              <span aria-hidden="true" className="text-slate-600 hidden lg:inline">
                ·
              </span>
              <span className="text-xs text-sky-300 font-mono-tabular truncate max-w-[220px] hidden lg:inline">
                Pieza: {inspectedPart.name}
              </span>
            </>
          )}
        </div>

        {/* Right: Expand Height Toggle + Collapse Dock Button */}
        <div className="flex items-center gap-2">
          {!isCollapsed && (
            <button
              onClick={() => setIsTallDock((prev) => !prev)}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-100 bg-[#111622] hover:bg-slate-800 border border-slate-600/90 rounded-lg shadow-xl transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            >
              {isTallDock ? (
                <>
                  <Minimize2 className="w-3.5 h-3.5 text-sky-400" />
                  Menú Compacto
                </>
              ) : (
                <>
                  <Maximize2 className="w-3.5 h-3.5 text-sky-400" />
                  Ampliar Menú
                </>
              )}
            </button>
          )}

          <button
            onClick={onToggleCollapse}
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 text-xs font-semibold text-slate-100 bg-[#111622] hover:bg-slate-800 border border-slate-600/90 rounded-lg shadow-xl transition-colors whitespace-nowrap shrink-0 cursor-pointer"
            title="Atajo de teclado: H"
          >
            {isCollapsed ? (
              <>
                <ChevronUp className="w-4 h-4 text-amber-400" />
                Mostrar Menús (H)
              </>
            ) : (
              <>
                <ChevronDown className="w-4 h-4 text-amber-400" />
                Ocultar Menús (H)
              </>
            )}
          </button>
        </div>
      </div>

      {/* Main High-Contrast Dock Container */}
      {!isCollapsed && (
        <div className="rounded-xl border-2 border-slate-700/90 bg-[#0f1420] shadow-2xl overflow-hidden">
          {/* Primary Navigation Tabs + Despiece + Mutar + Exportar */}
          <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 bg-[#161d2e] border-b border-slate-700">
          {/* High-Visibility Segmented Tabs */}
          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => onTabChange('catalog')}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'catalog'
                  ? 'bg-amber-500 text-slate-950 shadow-md'
                  : 'bg-[#0c101a] text-slate-100 border border-slate-700 hover:border-slate-500'
              }`}
            >
              <Grid className="w-4 h-4" />
              1. Personajes (40)
            </button>

            <button
              onClick={() => onTabChange('anatomy')}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'anatomy'
                  ? 'bg-amber-500 text-slate-950 shadow-md'
                  : 'bg-[#0c101a] text-slate-100 border border-slate-700 hover:border-slate-500'
              }`}
            >
              <Sliders className="w-4 h-4" />
              2. Anatomía y Cuerpo
            </button>

            <button
              onClick={() => onTabChange('equipment')}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'equipment'
                  ? 'bg-amber-500 text-slate-950 shadow-md'
                  : 'bg-[#0c101a] text-slate-100 border border-slate-700 hover:border-slate-500'
              }`}
            >
              <Palette className="w-4 h-4" />
              3. Equipo y Colores PBR
            </button>

            <button
              onClick={() => onTabChange('scene')}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'scene'
                  ? 'bg-amber-500 text-slate-950 shadow-md'
                  : 'bg-[#0c101a] text-slate-100 border border-slate-700 hover:border-slate-500'
              }`}
            >
              <Play className="w-4 h-4" />
              4. Animación y Cámara
            </button>

            <button
              onClick={() => onTabChange('rpg')}
              className={`inline-flex items-center gap-1.5 px-3 py-2 text-xs sm:text-sm font-semibold rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                activeTab === 'rpg'
                  ? 'bg-amber-500 text-slate-950 shadow-md'
                  : 'bg-[#0c101a] text-slate-100 border border-slate-700 hover:border-slate-500'
              }`}
            >
              <Shield className="w-4 h-4" />
              5. Stats RPG ({threatScore})
            </button>
          </div>

          {/* Right Action Controls */}
          <div className="flex items-center gap-2.5">
            <div className="flex items-center gap-2 text-xs text-slate-200 bg-[#0c101a] px-2.5 py-1.5 rounded-lg border border-slate-700">
              <Layers className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span className="whitespace-nowrap font-medium">Despiece:</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.01}
                value={explodedAmount}
                onChange={(e) => onExplodedChange(parseFloat(e.target.value))}
                className="w-20 accent-amber-500 cursor-pointer"
                aria-label="Vista explosionada de piezas"
              />
              <span className="font-mono-tabular text-xs text-amber-300 w-8">
                {Math.round(explodedAmount * 100)}%
              </span>
            </div>

            <button
              onClick={onRandomize}
              className="inline-flex items-center gap-1.5 px-3 py-2 text-xs font-semibold bg-slate-800 hover:bg-slate-700 text-white border border-slate-600 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer"
              title="Generar criatura híbrida aleatoria"
            >
              <Shuffle className="w-3.5 h-3.5 text-amber-400" />
              Aleatorio
            </button>

            <button
              onClick={onQuickExportGLB}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 text-xs font-bold bg-amber-500 hover:bg-amber-400 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer shadow-md"
              title="Descargar archivo binario .GLB"
            >
              <Download className="w-4 h-4" />
              Descargar .GLB
            </button>
          </div>
        </div>

        {/* Dock Body Content (Spacious & High Contrast) */}
        <div
          className={`p-4 overflow-y-auto transition-all ${
            isTallDock ? 'max-h-[420px]' : 'max-h-[260px]'
          }`}
        >
          {/* TAB 1: CATÁLOGO DE 30 PERSONAJES EN CUADRÍCULA VISIBLE */}
          {activeTab === 'catalog' && (
            <div className="space-y-3">
              {/* Category Filter Bar */}
              <div className="flex flex-wrap items-center justify-between gap-2 pb-2 border-b border-slate-700/80">
                <div className="flex flex-wrap items-center gap-1.5">
                  <button
                    onClick={() => setSelectedCategory('all')}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      selectedCategory === 'all'
                        ? 'bg-amber-500 text-slate-950'
                        : 'bg-[#171f30] text-slate-200 border border-slate-700 hover:border-slate-500'
                    }`}
                  >
                    Todos (40)
                  </button>
                  {(Object.keys(RACE_CATEGORY_LABELS) as RaceCategory[]).map((cat) => (
                    <button
                      key={cat}
                      onClick={() => setSelectedCategory(cat)}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        selectedCategory === cat
                          ? 'bg-amber-500 text-slate-950'
                          : 'bg-[#171f30] text-slate-200 border border-slate-700 hover:border-slate-500'
                      }`}
                    >
                      {RACE_CATEGORY_LABELS[cat]}
                    </button>
                  ))}
                </div>

                <span className="text-xs text-slate-300">
                  Haz clic en cualquier tarjeta para cargar el personaje al instante:
                </span>
              </div>

              {/* Multi-column Visible Grid of Characters */}
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2.5">
                {filteredArchetypes.map((archId) => {
                  const preset = ARCHETYPE_PRESETS[archId];
                  const isSelected = config.archetype === archId;
                  return (
                    <button
                      key={archId}
                      onClick={() => onChangeConfig(() => structuredClone(preset))}
                      className={`text-left p-2.5 rounded-lg border transition-all cursor-pointer flex flex-col justify-between gap-1.5 ${
                        isSelected
                          ? 'bg-amber-500/20 border-amber-400 ring-1 ring-amber-400 shadow-md'
                          : 'bg-[#161e2e] border-slate-700 hover:border-slate-500 hover:bg-[#1c263b]'
                      }`}
                    >
                      <div className="flex items-center justify-between gap-1">
                        <span
                          className={`text-xs font-bold truncate ${
                            isSelected ? 'text-amber-300' : 'text-slate-100'
                          }`}
                        >
                          {preset.name}
                        </span>
                        <span
                          className="w-3 h-3 rounded-sm shrink-0 border border-white/30"
                          style={{ backgroundColor: preset.materials.emissiveColor }}
                        />
                      </div>
                      <div className="text-[11px] text-slate-300 truncate">{preset.title}</div>
                      <div className="flex items-center justify-between text-[11px] text-slate-400 font-mono-tabular pt-0.5 border-t border-slate-700/60">
                        <span>{preset.stats.tier}</span>
                        <span className="text-amber-300">Nv.{preset.stats.level}</span>
                      </div>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB 2: ANATOMÍA, ESQUELETO Y MORFOLOGÍA */}
          {activeTab === 'anatomy' && (
            <div className="space-y-4">
              {/* Top Row: Skeleton Body Plan + Mesh Density (LOD) + Multi-Head Count */}
              <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-700/80">
                <label className="flex flex-wrap items-center gap-2 text-xs sm:text-sm font-medium text-slate-100">
                  <span>Esqueleto 3D:</span>
                  <select
                    value={config.bodyPlan}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        bodyPlan: e.target.value as BodyPlan,
                      }))
                    }
                    className="bg-[#171f30] border border-slate-600 rounded-lg px-2.5 py-1.5 text-xs sm:text-sm text-amber-300 font-semibold focus:outline-none focus:border-amber-400"
                  >
                    {BODY_PLAN_OPTIONS.map((bp) => (
                      <option key={bp.id} value={bp.id}>
                        {bp.label}
                      </option>
                    ))}
                  </select>
                </label>

                {/* Polygon Density (LOD) Selector */}
                <div className="flex flex-wrap items-center gap-1.5 text-xs font-medium text-slate-100 bg-[#161e2e] px-2.5 py-1.5 rounded-lg border border-slate-700">
                  <span className="text-slate-300 mr-1">Densidad Poligonal:</span>
                  {(
                    [
                      { id: 'low', label: 'Baja (Low-Poly + Texturas)' },
                      { id: 'medium', label: 'Media' },
                      { id: 'high', label: 'Alta' },
                    ] as Array<{ id: MeshDensity; label: string }>
                  ).map((lod) => {
                    const activeLod = getEffectiveMeshDensity(config);
                    return (
                      <button
                        key={lod.id}
                        onClick={() =>
                          onChangeConfig((prev) => ({
                            ...prev,
                            morphology: { ...prev.morphology, meshDensity: lod.id },
                          }))
                        }
                        className={`px-2.5 py-1 rounded font-semibold transition-colors cursor-pointer ${
                          activeLod === lod.id
                            ? 'bg-sky-400 text-slate-950'
                            : 'bg-[#0d121d] text-slate-200 border border-slate-600 hover:border-slate-400'
                        }`}
                      >
                        {lod.label}
                      </button>
                    );
                  })}
                </div>

                <div className="flex items-center gap-1.5 text-xs font-medium text-slate-100">
                  <span>Cabezas:</span>
                  {[1, 2, 3].map((hc) => (
                    <button
                      key={hc}
                      onClick={() =>
                        onChangeConfig((prev) => ({
                          ...prev,
                          morphology: { ...prev.morphology, headCount: hc },
                        }))
                      }
                      className={`px-2.5 py-1 rounded-md font-bold transition-colors cursor-pointer ${
                        (config.morphology.headCount || 1) === hc
                          ? 'bg-amber-500 text-slate-950'
                          : 'bg-[#171f30] text-slate-200 border border-slate-600 hover:border-slate-400'
                      }`}
                    >
                      {hc}x
                    </button>
                  ))}
                </div>
              </div>

              {/* Sliders Grid with High Contrast Labels */}
              <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-4 gap-4">
                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Altura Corporal</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.heightScale.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.75}
                    max={1.55}
                    step={0.02}
                    value={config.morphology.heightScale}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, heightScale: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Masa / Musculatura</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.torsoBulk.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.75}
                    max={1.6}
                    step={0.02}
                    value={config.morphology.torsoBulk}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, torsoBulk: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Largo de Cuello</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.neckLength.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.25}
                    max={1.85}
                    step={0.05}
                    value={config.morphology.neckLength}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, neckLength: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Largo de Cola</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.tailLength.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={1.9}
                    step={0.05}
                    value={config.morphology.tailLength}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, tailLength: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Fauces / Morro</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.snoutLength.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.3}
                    max={1.7}
                    step={0.05}
                    value={config.morphology.snoutLength}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, snoutLength: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Envergadura Alas</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.dorsalScale.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.5}
                    max={1.95}
                    step={0.05}
                    value={config.morphology.dorsalScale}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, dorsalScale: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Largo de Cuernos</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.hornLength.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.0}
                    max={1.85}
                    step={0.05}
                    value={config.morphology.hornLength}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, hornLength: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <div className="flex justify-between font-medium text-slate-200">
                    <span>Escala de Arma</span>
                    <span className="font-mono-tabular text-amber-300">
                      {config.morphology.weaponScale.toFixed(2)}x
                    </span>
                  </div>
                  <input
                    type="range"
                    min={0.65}
                    max={1.75}
                    step={0.05}
                    value={config.morphology.weaponScale}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        morphology: { ...prev.morphology, weaponScale: parseFloat(e.target.value) },
                      }))
                    }
                    className="accent-amber-500 cursor-pointer"
                  />
                </label>
              </div>
            </div>
          )}

          {/* TAB 3: PARTES, ARMAS Y MATERIALES PBR */}
          {activeTab === 'equipment' && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold flex items-center gap-1.5">
                    <Box className="w-4 h-4 text-amber-400" /> Cabeza / Fauces
                  </span>
                  <select
                    value={config.helm}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({ ...prev, helm: e.target.value as HelmStyle }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2.5 py-1.5 text-xs sm:text-sm text-white focus:outline-none focus:border-amber-400"
                  >
                    {HELM_OPTIONS.map((h) => (
                      <option key={h.id} value={h.id}>
                        {h.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold flex items-center gap-1.5">
                    <Sparkles className="w-4 h-4 text-purple-400" /> Alas / Dorsal
                  </span>
                  <select
                    value={config.dorsal}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({ ...prev, dorsal: e.target.value as DorsalType }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2.5 py-1.5 text-xs sm:text-sm text-white focus:outline-none focus:border-amber-400"
                  >
                    {DORSAL_OPTIONS.map((d) => (
                      <option key={d.id} value={d.id}>
                        {d.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold flex items-center gap-1.5">
                    <Swords className="w-4 h-4 text-amber-400" /> Arma / Ataque
                  </span>
                  <select
                    value={config.weapon}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({ ...prev, weapon: e.target.value as WeaponType }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2.5 py-1.5 text-xs sm:text-sm text-white focus:outline-none focus:border-amber-400"
                  >
                    {WEAPON_OPTIONS.map((w) => (
                      <option key={w.id} value={w.id}>
                        {w.label}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="flex flex-col gap-1.5 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold flex items-center gap-1.5">
                    <Shield className="w-4 h-4 text-sky-400" /> Mano Izquierda
                  </span>
                  <select
                    value={config.offhand}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        offhand: e.target.value as OffhandType,
                      }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2.5 py-1.5 text-xs sm:text-sm text-white focus:outline-none focus:border-amber-400"
                  >
                    {OFFHAND_OPTIONS.map((o) => (
                      <option key={o.id} value={o.id}>
                        {o.label}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {/* Surface Texture Pattern (Albedo + Bump) + Material Presets & Color Pickers */}
              <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-700">
                <div className="flex flex-wrap items-center gap-1.5 bg-[#161e2e] px-2.5 py-1.5 rounded-lg border border-slate-700">
                  <span className="text-xs font-semibold text-amber-300 mr-1">
                    Textura 3D:
                  </span>
                  {(
                    [
                      { id: 'scales', label: 'Escamas Dragón' },
                      { id: 'chitin_fur', label: 'Pelaje / Quitina' },
                      { id: 'damascus_plate', label: 'Acero Forjado' },
                      { id: 'runic_obsidian', label: 'Roca Volcánica' },
                      { id: 'none', label: 'Liso' },
                    ] as Array<{ id: TexturePatternId; label: string }>
                  ).map((tp) => {
                    const activeTex = getEffectiveTexturePattern(config);
                    return (
                      <button
                        key={tp.id}
                        onClick={() =>
                          onChangeConfig((prev) => ({
                            ...prev,
                            materials: { ...prev.materials, texturePattern: tp.id },
                          }))
                        }
                        className={`px-2.5 py-1 text-xs font-semibold rounded transition-colors cursor-pointer ${
                          activeTex === tp.id
                            ? 'bg-amber-500 text-slate-950'
                            : 'bg-[#0d121d] text-slate-200 border border-slate-600 hover:border-slate-400'
                        }`}
                      >
                        {tp.label}
                      </button>
                    );
                  })}

                  <div className="flex items-center gap-1.5 ml-2 pl-2 border-l border-slate-700 text-xs text-slate-200">
                    <span>Relieve:</span>
                    <input
                      type="range"
                      min={0}
                      max={1.4}
                      step={0.05}
                      value={config.materials.textureRelief ?? 0.75}
                      onChange={(e) =>
                        onChangeConfig((prev) => ({
                          ...prev,
                          materials: {
                            ...prev.materials,
                            textureRelief: parseFloat(e.target.value),
                          },
                        }))
                      }
                      className="w-16 accent-amber-500 cursor-pointer"
                    />
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-1.5">
                  {(Object.keys(MATERIAL_PRESETS) as MaterialPresetId[]).map((matId) => {
                    const p = MATERIAL_PRESETS[matId];
                    return (
                      <button
                        key={matId}
                        onClick={() =>
                          onChangeConfig((prev) => ({
                            ...prev,
                            materials: {
                              ...prev.materials,
                              preset: matId,
                              ...p,
                            },
                          }))
                        }
                        className={`inline-flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                          config.materials.preset === matId
                            ? 'bg-amber-500 text-slate-950 border-amber-400'
                            : 'bg-[#161e2e] border-slate-700 text-slate-200 hover:border-slate-500'
                        }`}
                      >
                        <span
                          className="w-3 h-3 rounded-sm border border-white/30"
                          style={{ backgroundColor: p.primaryColor }}
                        />
                        {p.label.split(' ')[0]} {p.label.split(' ')[1]}
                      </button>
                    );
                  })}
                </div>

                <div className="flex flex-wrap items-center gap-3 bg-[#161e2e] px-3 py-1.5 rounded-lg border border-slate-700">
                  <label className="flex items-center gap-1.5 text-xs font-medium text-slate-200 cursor-pointer">
                    <span>Primario</span>
                    <input
                      type="color"
                      value={config.materials.primaryColor}
                      onChange={(e) =>
                        onChangeConfig((prev) => ({
                          ...prev,
                          materials: { ...prev.materials, primaryColor: e.target.value },
                        }))
                      }
                      className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer"
                    />
                  </label>
                  <label className="flex items-center gap-1.5 text-xs font-medium text-slate-200 cursor-pointer">
                    <span>Secundario</span>
                    <input
                      type="color"
                      value={config.materials.secondaryColor}
                      onChange={(e) =>
                        onChangeConfig((prev) => ({
                          ...prev,
                          materials: { ...prev.materials, secondaryColor: e.target.value },
                        }))
                      }
                      className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer"
                    />
                  </label>
                  <label className="flex items-center gap-1.5 text-xs font-medium text-slate-200 cursor-pointer">
                    <span>Energía</span>
                    <input
                      type="color"
                      value={config.materials.emissiveColor}
                      onChange={(e) =>
                        onChangeConfig((prev) => ({
                          ...prev,
                          materials: { ...prev.materials, emissiveColor: e.target.value },
                        }))
                      }
                      className="w-6 h-6 rounded bg-transparent border-0 cursor-pointer"
                    />
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* TAB 4: ANIMACIÓN, CÁMARA Y ESCENA */}
          {activeTab === 'scene' && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
              <div className="bg-[#161e2e] p-3 rounded-lg border border-slate-700 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold text-slate-200">
                  <span>Estado Cinemático:</span>
                  <span className="font-mono-tabular text-amber-300">
                    Velocidad {animSpeed.toFixed(1)}x
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  {(
                    [
                      { id: 'idle', label: 'Reposo' },
                      { id: 'walk', label: 'Marcha / Vuelo' },
                      { id: 'attack', label: 'Ataque / Aliento' },
                      { id: 'hit', label: 'Impacto' },
                      { id: 'death', label: 'Derrota' },
                    ] as Array<{ id: AnimationState; label: string }>
                  ).map((st) => (
                    <button
                      key={st.id}
                      onClick={() => onAnimStateChange(st.id)}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        animState === st.id
                          ? 'bg-amber-500 text-slate-950 border-amber-400'
                          : 'bg-[#0d121d] border-slate-700 text-slate-200 hover:border-slate-500'
                      }`}
                    >
                      {st.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="bg-[#161e2e] p-3 rounded-lg border border-slate-700 space-y-2">
                <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <Camera className="w-4 h-4 text-sky-400" /> Ángulo de Cámara:
                </span>
                <div className="flex flex-wrap items-center gap-1.5">
                  {(
                    [
                      { id: 'hero_three_quarter', label: '3/4 Heroico' },
                      { id: 'front_ortho', label: 'Frontal' },
                      { id: 'weapon_closeup', label: 'Garras / Arma' },
                      { id: 'head_portrait', label: 'Cabeza / Fauces' },
                      { id: 'top_tactical', label: 'Cenital' },
                    ] as Array<{ id: CameraPresetId; label: string }>
                  ).map((cam) => (
                    <button
                      key={cam.id}
                      onClick={() => onCameraPresetChange(cam.id)}
                      className={`px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                        cameraPreset === cam.id
                          ? 'bg-sky-400 text-slate-950 border-sky-300'
                          : 'bg-[#0d121d] border-slate-700 text-slate-200 hover:border-slate-500'
                      }`}
                    >
                      {cam.label}
                    </button>
                  ))}
                </div>
              </div>

              <div className="bg-[#161e2e] p-3 rounded-lg border border-slate-700 space-y-2">
                <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <Sun className="w-4 h-4 text-amber-400" /> Iluminación y Hitboxes:
                </span>
                <div className="flex flex-wrap items-center gap-2">
                  <select
                    value={lightingPreset}
                    onChange={(e) => onLightingPresetChange(e.target.value as LightingPresetId)}
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-amber-400"
                  >
                    <option value="dark_forge">Forja Abisal</option>
                    <option value="studio_pbr">Estudio PBR Neutro</option>
                    <option value="frost_crypt">Cripta Glacial</option>
                    <option value="solar_arena">Arena Solar</option>
                  </select>

                  <button
                    onClick={onToggleHitboxes}
                    className={`inline-flex items-center gap-1 px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      showHitboxes
                        ? 'bg-emerald-400 text-slate-950 border-emerald-300'
                        : 'bg-[#0d121d] border-slate-700 text-slate-200'
                    }`}
                  >
                    <Eye className="w-3.5 h-3.5" />
                    Hitboxes
                  </button>

                  <button
                    onClick={onToggleWireframe}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      wireframe
                        ? 'bg-amber-400 text-slate-950 border-amber-300'
                        : 'bg-[#0d121d] border-slate-700 text-slate-200'
                    }`}
                  >
                    Malla
                  </button>

                  <button
                    onClick={onToggleAutoRotate}
                    className={`px-3 py-1.5 text-xs font-semibold rounded-md border transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                      autoRotate
                        ? 'bg-sky-400 text-slate-950 border-sky-300'
                        : 'bg-[#0d121d] border-slate-700 text-slate-200'
                    }`}
                  >
                    Girar
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* TAB 5: ATRIBUTOS RPG Y EXPORTACIÓN */}
          {activeTab === 'rpg' && (
            <div className="space-y-3">
              <div className="grid grid-cols-1 sm:grid-cols-3 md:grid-cols-6 gap-3 items-end">
                <label className="flex flex-col gap-1 text-xs sm:text-sm sm:col-span-2 bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold">Nombre del Personaje</span>
                  <input
                    type="text"
                    value={config.name}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        name: e.target.value,
                      }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2.5 py-1.5 text-xs sm:text-sm text-white focus:outline-none focus:border-amber-400"
                  />
                </label>

                <label className="flex flex-col gap-1 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold">Rango / Tier</span>
                  <select
                    value={config.stats.tier}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        stats: { ...prev.stats, tier: e.target.value as EnemyTier },
                      }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2 py-1.5 text-xs sm:text-sm text-white"
                  >
                    <option value="Minion">Minion</option>
                    <option value="Elite">Elite</option>
                    <option value="Mini-Boss">Mini-Boss</option>
                    <option value="Raid Boss">Raid Boss</option>
                  </select>
                </label>

                <label className="flex flex-col gap-1 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold">Elemento</span>
                  <select
                    value={config.stats.element}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        stats: { ...prev.stats, element: e.target.value as ElementAffinity },
                      }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2 py-1.5 text-xs sm:text-sm text-white"
                  >
                    <option value="Fuego Dracónico">Fuego Dracónico</option>
                    <option value="Abisal">Abisal</option>
                    <option value="Fuego Solar">Fuego Solar</option>
                    <option value="Escarcha">Escarcha</option>
                    <option value="Veneno">Veneno</option>
                    <option value="Arcano">Arcano</option>
                    <option value="Salvaje / Físico">Salvaje / Físico</option>
                  </select>
                </label>

                <label className="flex flex-col gap-1 text-xs sm:text-sm bg-[#161e2e] p-2.5 rounded-lg border border-slate-700">
                  <span className="text-slate-200 font-semibold">IA de Combate</span>
                  <select
                    value={config.stats.aiBehavior}
                    onChange={(e) =>
                      onChangeConfig((prev) => ({
                        ...prev,
                        stats: { ...prev.stats, aiBehavior: e.target.value as AiBehaviorType },
                      }))
                    }
                    className="bg-[#0d121d] border border-slate-600 rounded-md px-2 py-1.5 text-xs sm:text-sm text-white"
                  >
                    <option value="Aerial Breath Sweeper">Barrido Aéreo / Aliento</option>
                    <option value="Pack Alpha Predator">Depredador Alfa</option>
                    <option value="Aggressive Berserker">Berserker Agresivo</option>
                    <option value="Shield Wall Tank">Muro Defensivo</option>
                    <option value="Arcane Artillery">Artillería Arcana</option>
                    <option value="Ambush Stalker">Acechador Emboscada</option>
                    <option value="Boss Multi-Phase">Jefe Multifase</option>
                  </select>
                </label>

                <button
                  onClick={onOpenExportModal}
                  className="h-full inline-flex items-center justify-center gap-1.5 px-4 py-3 text-xs sm:text-sm font-bold bg-sky-400 hover:bg-sky-300 text-slate-950 rounded-lg transition-colors whitespace-nowrap shrink-0 cursor-pointer shadow-md"
                >
                  <Code2 className="w-4 h-4" />
                  Código Three.js / GLB
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      )}
    </div>
  );
};
