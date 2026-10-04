import React, { useEffect, useRef } from "react";
import * as THREE from "three";

export default function FetusModel3D({ week = 28, className = "" }) {
  const mountRef = useRef(null);
  const animFrameRef = useRef(null);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 200;
    const height = container.clientHeight || 200;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(48, width / height, 0.1, 100);
    camera.position.set(0, 0, 3.4);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(width, height);
    renderer.setClearColor(0x000000, 0);
    container.appendChild(renderer.domElement);

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffd6e0, 1.4);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffe0ec, 2.8);
    keyLight.position.set(2, 3, 2);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0xd0c2f0, 1.6);
    fillLight.position.set(-2, -1, 2);
    scene.add(fillLight);

    // Materials
    const skinMat = new THREE.MeshPhongMaterial({
      color: 0xffd6c8,
      shininess: 55,
    });
    const detailMat = new THREE.MeshPhongMaterial({
      color: 0xf5a0b8,
      shininess: 30,
    });
    const fluidMat = new THREE.MeshPhongMaterial({
      color: 0xffd6e0,
      transparent: true,
      opacity: 0.28,
    });
    const cordMat = new THREE.MeshPhongMaterial({
      color: 0xe8a0b0,
      shininess: 35,
    });
    const glowMat = new THREE.MeshPhongMaterial({
      color: 0xe05675,
      emissive: 0xe05675,
      emissiveIntensity: 0.5,
    });
    const ringMat = new THREE.MeshPhongMaterial({
      color: 0xe05675,
      transparent: true,
      opacity: 0.35,
    });

    const fetusGroup = new THREE.Group();

    // Amniotic sac representation
    const sacGeo = new THREE.SphereGeometry(1.08, 32, 32);
    fetusGroup.add(new THREE.Mesh(sacGeo, fluidMat));

    // Baby Body
    const bodyGeo = new THREE.SphereGeometry(0.52, 32, 32);
    bodyGeo.scale(1.0, 1.25, 0.9);
    const bodyMesh = new THREE.Mesh(bodyGeo, skinMat);
    bodyMesh.position.set(0.08, -0.05, 0);
    fetusGroup.add(bodyMesh);

    // Baby Head
    const headGeo = new THREE.SphereGeometry(0.38, 32, 32);
    const headMesh = new THREE.Mesh(headGeo, skinMat);
    headMesh.position.set(-0.24, 0.52, 0);
    fetusGroup.add(headMesh);

    // Eyes
    ["L", "R"].forEach((side, i) => {
      const eyeGeo = new THREE.SphereGeometry(0.062, 16, 8);
      eyeGeo.scale(1.0, 0.2, 0.55);
      const eyeMesh = new THREE.Mesh(eyeGeo, detailMat);
      eyeMesh.position.set(-0.24 + i * 0.26 - 0.13, 0.55, 0.33);
      fetusGroup.add(eyeMesh);
    });

    // Nose
    const noseMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.038, 8, 8),
      detailMat
    );
    noseMesh.position.set(-0.24, 0.46, 0.37);
    fetusGroup.add(noseMesh);

    // Arm
    const armMesh = new THREE.Mesh(
      new THREE.CylinderGeometry(0.1, 0.08, 0.44, 16),
      skinMat
    );
    armMesh.position.set(-0.42, 0.04, 0.31);
    armMesh.rotation.set(-0.4, 0, Math.PI / 3);
    fetusGroup.add(armMesh);

    // Hand
    const handMesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.1, 16, 16),
      skinMat
    );
    handMesh.position.set(-0.35, 0.25, 0.4);
    fetusGroup.add(handMesh);

    // Legs
    [
      [-0.18, -0.52, 0.18, 0.6, -0.7],
      [0.28, -0.52, 0.15, -0.5, -0.8],
    ].forEach(([x, y, z, rz, rx]) => {
      const legMesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.12, 0.09, 0.5, 16),
        skinMat
      );
      legMesh.position.set(x, y, z);
      legMesh.rotation.set(rx, 0, rz);
      fetusGroup.add(legMesh);
    });

    // Umbilical Cord
    const cordPath = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.1, -0.1, 0.2),
      new THREE.Vector3(0.32, 0.1, 0.3),
      new THREE.Vector3(0.52, -0.05, 0.1),
    ]);
    const cordGeo = new THREE.TubeGeometry(cordPath, 20, 0.024, 8, false);
    fetusGroup.add(new THREE.Mesh(cordGeo, cordMat));

    // Glowing vital particles orbiting
    const glowSpheres = Array.from({ length: 6 }, (_, i) => {
      const s = new THREE.Mesh(
        new THREE.SphereGeometry(0.038, 8, 8),
        glowMat.clone()
      );
      const angle = (i / 6) * Math.PI * 2;
      s.position.set(Math.cos(angle) * 0.8, Math.sin(angle) * 0.58, 0.28);
      s.userData = {
        angle,
        speed: 0.5 + Math.random() * 0.5,
        phase: i,
      };
      fetusGroup.add(s);
      return s;
    });

    // Orbiting subtle energy ring
    const ringGeo = new THREE.TorusGeometry(0.94, 0.014, 8, 64);
    const ringMesh = new THREE.Mesh(ringGeo, ringMat);
    fetusGroup.add(ringMesh);

    scene.add(fetusGroup);

    let t = 0;
    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      t += 0.011;

      fetusGroup.rotation.y = Math.sin(t * 0.38) * 0.22;
      fetusGroup.rotation.x = Math.sin(t * 0.22) * 0.07;

      // Gentle fetal breathing motion
      const breath = 1 + Math.sin(t * 4.1) * 0.024;
      bodyMesh.scale.set(breath, breath * 1.25, breath * 0.9);

      // Particle orbital animation
      glowSpheres.forEach((s) => {
        s.userData.angle += s.userData.speed * 0.008;
        s.position.x =
          Math.cos(s.userData.angle) *
          (0.74 + Math.sin(t * 1.9 + s.userData.phase) * 0.09);
        s.position.y =
          Math.sin(s.userData.angle) *
          (0.55 + Math.cos(t * 1.4 + s.userData.phase) * 0.07);
        s.material.emissiveIntensity =
          0.25 + Math.sin(t * 4.1 + s.userData.phase) * 0.28;
        s.scale.setScalar(0.6 + Math.sin(t * 2.9 + s.userData.phase) * 0.3);
      });

      ringMesh.scale.setScalar(1 + Math.sin(t * 1.9) * 0.04);
      ringMat.opacity = 0.22 + Math.sin(t * 1.9) * 0.14;

      renderer.render(scene, camera);
    };

    animate();

    const handleResize = () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    };

    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      cancelAnimationFrame(animFrameRef.current);
      renderer.dispose();
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
    };
  }, [week]);

  return <div ref={mountRef} className={`w-full h-full ${className}`} />;
}
