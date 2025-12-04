// Cylinder: 10 mm × 200 mm
// 单位默认是毫米 (mm)

$fa = 1;            // 细分角度；越小模型越光滑
$fs = 0.5;          // 最小边长，同样影响光滑度

cylinder(h = 200,   // 高度 200 mm = 20 cm
         d = 10,    // 直径 10 mm = 1 cm
         center = false);  // 底面位于 Z=0，顶面在 Z=200