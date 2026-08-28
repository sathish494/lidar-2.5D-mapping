/**
 * FoveaMap Main Dashboard Application Controller
 */

document.addEventListener('DOMContentLoaded', async () => {
  const renderer = new FoveaRenderer3D('canvas3d');
  const metricsHUD = new MetricsHUDController();
  const scenarioMgr = new ScenarioManager();

  let isPlaying = true;
  let isLiveWS = false;
  let ws = null;
  let playbackInterval = null;
  let fpsRate = 10;

  // Live Foveation Overrides
  let overrideParams = {
    speed_mps: 8.0,
    steering_angle_rad: 0.0,
    base_fine_radius_m: 10.0,
    max_stretch: 2.5,
    shear_strength: 0.6,
  };

  // DOM Elements
  const selectScenario = document.getElementById('select-scenario');
  const selectMode = document.getElementById('select-mode');
  const btnPlay = document.getElementById('btn-play');
  const btnStep = document.getElementById('btn-step');
  const btnReset = document.getElementById('btn-reset');
  const sliderScrub = document.getElementById('slider-scrub');
  const lblFrame = document.getElementById('lbl-frame-counter');
  const lblScenarioDesc = document.getElementById('lbl-scenario-desc');

  // Sliders
  const sliderSpeed = document.getElementById('slider-speed');
  const valSpeed = document.getElementById('val-speed');
  const sliderSteering = document.getElementById('slider-steering');
  const valSteering = document.getElementById('val-steering');
  const sliderMaxStretch = document.getElementById('slider-max-stretch');
  const valMaxStretch = document.getElementById('val-max-stretch');
  const sliderShear = document.getElementById('slider-shear');
  const valShear = document.getElementById('val-shear');
  const sliderBaseRadius = document.getElementById('slider-base-radius');
  const valBaseRadius = document.getElementById('val-base-radius');

  // Camera Buttons
  document.getElementById('btn-view-perspective')?.addEventListener('click', () => renderer.setCameraView('perspective'));
  document.getElementById('btn-view-topdown')?.addEventListener('click', () => renderer.setCameraView('topdown'));
  document.getElementById('btn-view-cockpit')?.addEventListener('click', () => renderer.setCameraView('cockpit'));

  // Slider Event Listeners (Live Dynamic Foveation updates)
  function onSliderChange() {
    overrideParams.speed_mps = parseFloat(sliderSpeed.value);
    valSpeed.textContent = `${overrideParams.speed_mps.toFixed(1)} m/s`;

    const steerDeg = parseFloat(sliderSteering.value);
    overrideParams.steering_angle_rad = (steerDeg * Math.PI) / 180.0;
    valSteering.textContent = `${steerDeg > 0 ? '+' : ''}${steerDeg.toFixed(0)}°`;

    overrideParams.max_stretch = parseFloat(sliderMaxStretch.value);
    valMaxStretch.textContent = `${overrideParams.max_stretch.toFixed(1)}x`;

    overrideParams.shear_strength = parseFloat(sliderShear.value);
    valShear.textContent = overrideParams.shear_strength.toFixed(2);

    overrideParams.base_fine_radius_m = parseFloat(sliderBaseRadius.value);
    valBaseRadius.textContent = `${overrideParams.base_fine_radius_m.toFixed(1)} m`;

    // Immediately update 3D contour
    renderer.polarOverlay.updateFoveationContour(
      overrideParams.speed_mps,
      overrideParams.steering_angle_rad,
      overrideParams.base_fine_radius_m,
      overrideParams.max_stretch,
      overrideParams.shear_strength
    );

    // If WebSocket is active, send live params
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({
        command: 'set_params',
        speed_mps: overrideParams.speed_mps,
        steering_angle_rad: overrideParams.steering_angle_rad,
      }));
    }
  }

  [sliderSpeed, sliderSteering, sliderMaxStretch, sliderShear, sliderBaseRadius].forEach((s) => {
    s?.addEventListener('input', onSliderChange);
  });

  // Scenario Change Handler
  async function switchScenario(scenarioId) {
    const frames = await scenarioMgr.loadScenario(scenarioId);
    if (sliderScrub) {
      sliderScrub.max = Math.max(0, frames.length - 1);
      sliderScrub.value = 0;
    }
    const currentFrame = scenarioMgr.getCurrentFrame();
    if (currentFrame) {
      if (lblScenarioDesc) lblScenarioDesc.textContent = currentFrame.description || '';
      renderCurrentFrame(currentFrame);
    }
  }

  selectScenario?.addEventListener('change', (e) => {
    switchScenario(e.target.value);
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ command: 'set_scenario', scenario_id: e.target.value }));
    }
  });

  // Playback Control Handlers
  btnPlay?.addEventListener('click', () => {
    isPlaying = !isPlaying;
    btnPlay.textContent = isPlaying ? '⏸ Pause' : '▶ Play';
  });

  btnStep?.addEventListener('click', () => {
    isPlaying = false;
    if (btnPlay) btnPlay.textContent = '▶ Play';
    const f = scenarioMgr.nextFrame();
    if (f) renderCurrentFrame(f);
  });

  btnReset?.addEventListener('click', () => {
    const f = scenarioMgr.setFrame(0);
    if (f) renderCurrentFrame(f);
  });

  sliderScrub?.addEventListener('input', (e) => {
    isPlaying = false;
    if (btnPlay) btnPlay.textContent = '▶ Play';
    const f = scenarioMgr.setFrame(parseInt(e.target.value));
    if (f) renderCurrentFrame(f);
  });

  function renderCurrentFrame(frameData) {
    if (!frameData) return;

    // Apply live foveation slider overrides if adjusted by user
    frameData.foveation_params = {
      base_fine_radius_m: overrideParams.base_fine_radius_m,
      max_stretch: overrideParams.max_stretch,
      shear_strength: overrideParams.shear_strength,
    };

    // Update 3D viewport
    renderer.renderFrame(frameData);

    // Update Telemetry HUD
    metricsHUD.updateMetrics(frameData);

    // Update Scrub bar and label
    if (sliderScrub && !isLiveWS) {
      sliderScrub.value = scenarioMgr.currentFrameIdx;
    }
    if (lblFrame) {
      const total = scenarioMgr.frames.length || 30;
      lblFrame.textContent = `Frame ${scenarioMgr.currentFrameIdx + 1} / ${total}`;
    }
  }

  // Precomputed Playback Loop (10 Hz)
  function startPlaybackLoop() {
    if (playbackInterval) clearInterval(playbackInterval);
    playbackInterval = setInterval(() => {
      if (isPlaying && !isLiveWS) {
        const frame = scenarioMgr.nextFrame();
        if (frame) renderCurrentFrame(frame);
      }
    }, 1000 / fpsRate);
  }

  // WebSocket Live Mode
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/grid`;
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      console.log('[WS] Connected to FoveaMap live stream.');
      document.getElementById('ws-status-dot')?.style.setProperty('background-color', '#22c55e');
    };

    ws.onmessage = (event) => {
      if (isLiveWS) {
        const frameData = JSON.parse(event.data);
        renderCurrentFrame(frameData);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected. Reconnecting in 3s...');
      document.getElementById('ws-status-dot')?.style.setProperty('background-color', '#ef4444');
      setTimeout(initWebSocket, 3000);
    };
  }

  selectMode?.addEventListener('change', (e) => {
    isLiveWS = e.target.value === 'live_ws';
    if (isLiveWS && !ws) initWebSocket();
  });

  // Initial Load
  await switchScenario('synthetic_kitti_like');
  startPlaybackLoop();
  initWebSocket();
});
