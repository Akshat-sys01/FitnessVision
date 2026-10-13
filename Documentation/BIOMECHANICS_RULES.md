# FitnessVision Biomechanics & Heuristic Rules Reference

## 1. Mathematical Formulas

### 1.1 Joint Angles
Given 3 planar landmarks $A$, $B$ (vertex), and $C$:
$$\vec{u} = A - B$$
$$\vec{v} = C - B$$
$$\theta = \arccos\left(\text{clamp}\left(\frac{\vec{u} \cdot \vec{v}}{\|\vec{u}\| \|\vec{v}\|}, -1.0, 1.0\right)\right) \times \frac{180}{\pi}$$

### 1.2 Torso Inclination
Given shoulder midpoint $S$ and hip midpoint $H$:
$$dx = |S_x - H_x|$$
$$dy = |S_y - H_y|$$
$$\alpha = \arctan\left(\frac{dx}{dy}\right) \times \frac{180}{\pi}$$
Where $\alpha = 0^\circ$ corresponds to a strictly vertical torso posture.

### 1.3 Bilateral Symmetry
Given left angle $\theta_L$ and right angle $\theta_R$:
$$\Delta_{\text{abs}} = |\theta_L - \theta_R|$$
$$\Delta_{\%} = \frac{|\theta_L - \theta_R|}{(\theta_L + \theta_R) / 2} \times 100$$

---

## 2. Threshold Configurations

| Metric | Threshold | Condition Flagged | Actionable Feedback |
| :--- | :--- | :--- | :--- |
| Squat Torso Inclination | $> 48.0^\circ$ | `EXCESSIVE_FORWARD_LEAN` | "Torso is leaning too far forward. Keep your chest elevated." |
| Squat Target Depth | $\le 100.0^\circ$ | Bottom Reached | "Good depth and control. Drive back up!" |
| Squat Reversal Depth | $> 115.0^\circ$ | `INSUFFICIENT_SQUAT_DEPTH` | "Squat depth was shallow. Lower hips until thighs are parallel." |
| Squat Knee Inward Ratio | $< 0.65 \times \text{ankles}$ | `KNEE_VALGUS` | "Knees are caving inward. Push your knees outward over toes." |
| Bicep Curl Upper Arm Flare | $> 40.0^\circ$ | `EXCESSIVE_ELBOW_DRIFT` | "Elbows are drifting or swinging. Keep upper arms pinned at sides." |
| Bicep Curl Torso Sway | $> 22.0^\circ$ | `TORSO_SWAY` | "Torso is swinging backward. Stand still and brace your core." |
| Bicep Curl Flexion Top | $\le 65.0^\circ$ | Top Flexion Reached | "Full contraction reached. Lower under control." |
| Bicep Curl Extension Bottom | $\ge 145.0^\circ$ | Repetition Completed | "Complete extension achieved." |
