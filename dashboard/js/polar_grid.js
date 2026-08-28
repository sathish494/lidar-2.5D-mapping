/**
 * FoveaMap Polar Grid & Dynamic Foveation Contour Overlay for Three.js
 */

class PolarGridOverlay {
  constructor(scene) {
    this.scene = scene;
    this.gridGroup = new THREE.Group();
    this.scene.add(this.gridGroup);

    this.foveaLine = null;
    this.foveaMesh = null;
    this.initStaticRings();
    this.initFoveationMesh();
  }

  initStaticRings() {
    // Static concentric guide rings at 10m, 30m, 50m, 80m, 100m
    const ringRadii = [10.0, 30.0, 50.0, 80.0, 100.0];
    const ringMaterial = new THREE.LineBasicMaterial({
      color: 0x334155,
      transparent: true,
      opacity: 0.4,
      linewidth: 1,
    });

    ringRadii.forEach((r) => {
      const circleGeo = new THREE.BufferGeometry();
      const points = [];
      const segments = 128;
      for (let i = 0; i <= segments; i++) {
        const theta = (i / segments) * Math.PI * 2;
        points.push(new THREE.Vector3(r * Math.cos(theta), r * Math.sin(theta), -1.55));
      }
      circleGeo.setFromPoints(points);
      const line = new THREE.Line(circleGeo, ringMaterial);
      this.gridGroup.add(line);
    });

    // Cross axes (X and Y forward/lateral)
    const axesGeo = new THREE.BufferGeometry();
    const axisPts = [
      new THREE.Vector3(-100, 0, -1.55),
      new THREE.Vector3(100, 0, -1.55),
      new THREE.Vector3(0, -100, -1.55),
      new THREE.Vector3(0, 100, -1.55),
    ];
    axesGeo.setFromPoints(axisPts);
    const axesLine = new THREE.LineSegments(axesGeo, ringMaterial);
    this.gridGroup.add(axesLine);

    // Ego vehicle marker (Green/Cyan Arrow at origin)
    const arrowGeo = new THREE.BufferGeometry();
    const arrowPts = [
      new THREE.Vector3(2.0, 0.0, -1.45),
      new THREE.Vector3(-1.0, -0.8, -1.45),
      new THREE.Vector3(-0.5, 0.0, -1.45),
      new THREE.Vector3(-1.0, 0.8, -1.45),
      new THREE.Vector3(2.0, 0.0, -1.45),
    ];
    arrowGeo.setFromPoints(arrowPts);
    const arrowMat = new THREE.LineBasicMaterial({ color: 0x06b6d4, linewidth: 2 });
    const arrow = new THREE.Line(arrowGeo, arrowMat);
    this.gridGroup.add(arrow);
  }

  initFoveationMesh() {
    // Dynamic Foveation boundary contour line
    const segments = 120;
    const points = [];
    for (let i = 0; i <= segments; i++) {
      points.push(new THREE.Vector3(0, 0, -1.48));
    }
    const lineGeo = new THREE.BufferGeometry().setFromPoints(points);
    const lineMat = new THREE.LineBasicMaterial({
      color: 0x06b6d4,
      transparent: true,
      opacity: 0.9,
      linewidth: 2,
    });
    this.foveaLine = new THREE.Line(lineGeo, lineMat);
    this.gridGroup.add(this.foveaLine);
  }

  updateFoveationContour(speed_mps, steering_angle_rad, base_radius = 10.0, max_stretch = 2.5, shear_strength = 0.6) {
    if (!this.foveaLine) return;

    const segments = 120;
    const positions = this.foveaLine.geometry.attributes.position.array;

    // Calculate speed stretch
    const speed_ref = 20.0;
    const ratio = Math.min(Math.max(0.0, speed_mps) / speed_ref, 1.0);
    const stretch = 1.0 + (max_stretch - 1.0) * ratio;

    // Steering norm
    const steer_norm = Math.max(-1.0, Math.min(1.0, steering_angle_rad / Math.PI));

    for (let i = 0; i <= segments; i++) {
      const theta = (i / segments) * Math.PI * 2 - Math.PI;

      // Forward hemisphere gets speed stretch
      const forward_bias = Math.abs(theta) <= Math.PI / 2.0 ? stretch : 1.0;
      const alignment = Math.cos(theta - steering_angle_rad);
      const lateral_bias = 1.0 + shear_strength * steer_norm * alignment;

      const r = Math.max(1.0, base_radius * forward_bias * lateral_bias);

      const x = r * Math.cos(theta);
      const y = r * Math.sin(theta);
      const z = -1.48;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    this.foveaLine.geometry.attributes.position.needsUpdate = true;
  }
}

window.PolarGridOverlay = PolarGridOverlay;
