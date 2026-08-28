/**
 * Scenario Management & Data Fetching for FoveaMap Dashboard
 */

class ScenarioManager {
  constructor() {
    this.currentScenarioId = 'synthetic_kitti_like';
    this.frames = [];
    this.currentFrameIdx = 0;
  }

  async loadScenario(scenarioId) {
    this.currentScenarioId = scenarioId;
    this.currentFrameIdx = 0;

    try {
      const response = await fetch(`/api/frames/${scenarioId}`);
      if (!response.ok) {
        throw new Error(`HTTP error ${response.status}`);
      }
      this.frames = await response.json();
      return this.frames;
    } catch (err) {
      console.warn(`[ScenarioManager] Could not fetch from server (${err}). Using fallback procedural data.`);
      return [];
    }
  }

  getCurrentFrame() {
    if (!this.frames || this.frames.length === 0) return null;
    return this.frames[this.currentFrameIdx];
  }

  nextFrame() {
    if (!this.frames || this.frames.length === 0) return null;
    this.currentFrameIdx = (this.currentFrameIdx + 1) % this.frames.length;
    return this.frames[this.currentFrameIdx];
  }

  setFrame(idx) {
    if (!this.frames || this.frames.length === 0) return null;
    this.currentFrameIdx = Math.max(0, Math.min(idx, this.frames.length - 1));
    return this.frames[this.currentFrameIdx];
  }
}

window.ScenarioManager = ScenarioManager;
