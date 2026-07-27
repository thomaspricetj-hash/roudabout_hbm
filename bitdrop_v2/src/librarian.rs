// ============================================================
// Librarian + Assistant
// - Lazy activation based on PatternLibrary + GSM
// - Produces tuning decisions for the BitDrop engine
// ============================================================

use crate::pattern_library::{GlobalStructureMap, PatternAnalysis, PatternLibrary};

// ----------------------------------------------------------
// Minimal payload stats mirror (engine can construct this)
// ----------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct PayloadStats {
    pub len: usize,
    pub entropy: f64,
    pub tier: u8,
}

// ----------------------------------------------------------
// Librarian decision: how to tune the engine
// ----------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct LibrarianDecision {
    pub use_light_bd3d: bool,
    pub skip_pts: bool,
    pub skip_semantic: bool,
    pub prefer_rle_fastpath: bool,
    pub max_layers_cap: Option<u16>,
    pub block_depth_scale: f32, // 1.0 = unchanged, >1.0 = deeper blocks
}

impl LibrarianDecision {
    pub fn neutral() -> Self {
        Self {
            use_light_bd3d: false,
            skip_pts: false,
            skip_semantic: false,
            prefer_rle_fastpath: false,
            max_layers_cap: None,
            block_depth_scale: 1.0,
        }
    }
}

// ----------------------------------------------------------
// Librarian Assistant (fine-grained heuristics)
// ----------------------------------------------------------
#[derive(Debug)]
pub struct LibrarianAssistant;

impl LibrarianAssistant {
    pub fn new() -> Self {
        Self
    }

    pub fn refine_decision(
        &self,
        stats: &PayloadStats,
        gsm: &GlobalStructureMap,
        matched_name: Option<&str>,
        mut decision: LibrarianDecision,
    ) -> LibrarianDecision {
        // Strong repetitive structure → RLE + shallow BD3D
        if gsm.repeat_ratio > 0.5 || gsm.longest_run > 256 {
            decision.prefer_rle_fastpath = true;
            decision.use_light_bd3d = true;
            decision.skip_pts = true;
            decision.skip_semantic = true;
            decision.max_layers_cap = Some(1);
            decision.block_depth_scale = 2.0;
        }

        // Medium repetition but not extreme → light BD3D, skip PTS
        if !decision.prefer_rle_fastpath && gsm.repeat_ratio > 0.25 {
            decision.use_light_bd3d = true;
            decision.skip_pts = true;
            if decision.max_layers_cap.is_none() {
                decision.max_layers_cap = Some(2);
            }
            if decision.block_depth_scale < 1.5 {
                decision.block_depth_scale = 1.5;
            }
        }

        // Very high entropy → avoid over-engineering
        if stats.entropy > 7.2 {
            decision.skip_pts = true;
            decision.skip_semantic = true;
            decision.use_light_bd3d = false;
            decision.prefer_rle_fastpath = false;
            decision.max_layers_cap = decision.max_layers_cap.or(Some(2));
        }

        // Very small payloads → don't bother with heavy machinery
        if stats.len < 64 * 1024 {
            decision.skip_pts = true;
            decision.skip_semantic = true;
            decision.use_light_bd3d = true;
            decision.max_layers_cap = decision.max_layers_cap.or(Some(1));
        }

        // Pattern-specific tweaks (you can extend this table)
        if let Some(name) = matched_name {
            match name {
                "highly_repetitive_stream" => {
                    decision.prefer_rle_fastpath = true;
                    decision.use_light_bd3d = true;
                    decision.skip_pts = true;
                    decision.skip_semantic = true;
                    decision.max_layers_cap = Some(1);
                    decision.block_depth_scale = 2.5;
                }
                "grid_like_numeric" => {
                    decision.use_light_bd3d = true;
                    decision.skip_pts = false;
                    decision.skip_semantic = false;
                    decision.max_layers_cap = decision.max_layers_cap.or(Some(2));
                    if decision.block_depth_scale < 1.3 {
                        decision.block_depth_scale = 1.3;
                    }
                }
                "medium_entropy_blocky" => {
                    decision.use_light_bd3d = true;
                    decision.skip_pts = false;
                    decision.skip_semantic = false;
                    decision.max_layers_cap = decision.max_layers_cap.or(Some(3));
                }
                _ => {}
            }
        }

        decision
    }
}

// ----------------------------------------------------------
// Librarian (lazy activation)
// ----------------------------------------------------------
#[derive(Debug)]
pub struct Librarian {
    library: PatternLibrary,
    assistant: LibrarianAssistant,
}

impl Librarian {
    pub fn new_with_default_library() -> Self {
        Self {
            library: PatternLibrary::with_default_patterns(),
            assistant: LibrarianAssistant::new(),
        }
    }

    pub fn library(&self) -> &PatternLibrary {
        &self.library
    }

    pub fn library_mut(&mut self) -> &mut PatternLibrary {
        &mut self.library
    }

    /// Analyze payload + stats and produce a tuning decision.
    /// This is cheap enough to call once per encode.
    pub fn advise(&self, payload: &[u8], stats: &PayloadStats) -> LibrarianDecision {
        if payload.is_empty() {
            return LibrarianDecision::neutral();
        }

        // Analyze structure + pattern match
        let analysis: PatternAnalysis = self.library.analyze(payload, stats.entropy);
        let gsm = analysis.gsm;

        // Lazy activation: if structure is weak and no match, stay neutral
        let strong_structure =
            gsm.repeat_ratio > 0.15 || gsm.longest_run > 32 || analysis.matched.is_some();

        if !strong_structure {
            return LibrarianDecision::neutral();
        }

        let matched_name = analysis.matched.as_ref().map(|e| e.name);
        let base_decision = LibrarianDecision::neutral();
        self.assistant
            .refine_decision(stats, &gsm, matched_name, base_decision)
    }
}
