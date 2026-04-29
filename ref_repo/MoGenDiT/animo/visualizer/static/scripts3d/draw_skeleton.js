import * as THREE from 'three';
// 请帮我实现使用three.js可视化骨架的函数，我的输入是一个骨架的每个关键点的坐标，我希望可视化两个东西：1. 在每个关键点的位置处，可视化一个球的mesh；2. 在edge里包含的每条边，可视化一个四棱锥的mesh，这个四棱锥的长度是这条边的长度，四棱锥的方向和这条边的方向一致
function drawSkeleton(keypoints, edges, scene, radius_joint = 0.015, radius_limb = 0.015) {
    // ========== 1. 声明局部变量，避免全局污染 ==========
    let geometries = [];

    // ========== 2. 异常处理：校验输入有效性 ==========
    if (!scene || !Array.isArray(keypoints) || !Array.isArray(edges)) {
        console.error('无效输入：scene/keypoints/edges必须为有效值');
        return;
    }

    // ========== 3. 清理旧骨架（核心：避免内存泄漏+重复渲染） ==========
    // 遍历场景，移除所有标记为骨架的元素
    scene.children.forEach(child => {
        if (child.userData.isSkeletonElement) {
            scene.remove(child);
            // 释放几何/材质内存
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
    });

    // ========== 4. 复用材质（优化性能） ==========
    const jointMaterial = new THREE.MeshStandardMaterial({ color: 0x6600ff });
    const limbMaterial = new THREE.MeshStandardMaterial({ color: 0x9999ff });

    // ========== 5. 绘制关节（球体） ==========
    keypoints.forEach((point, idx) => {
        // 容错：确保point是3维数组
        if (!Array.isArray(point) || point.length < 3) {
            console.warn(`无效的关键点${idx}：`, point);
            return;
        }
        const [x, y, z] = point;
        const sphereGeometry = new THREE.SphereGeometry(radius_joint, 16, 16); // 降低分段数，提升性能
        const sphere = new THREE.Mesh(sphereGeometry, jointMaterial);
        sphere.position.set(x, y, z);
        sphere.userData.isSkeletonElement = true; // 标记为骨架元素，方便清理
        geometries.push(sphere);
        scene.add(sphere);
    });

    // 绘制骨骼（四棱锥，核心修复：消除重叠）
    edges.forEach((edge, edgeIdx) => {
        const [startIdx, endIdx] = edge;
        if (startIdx >= keypoints.length || endIdx >= keypoints.length) {
            console.warn(`无效的边${edgeIdx}：索引超出关键点范围`);
            return;
        }
        const start = keypoints[startIdx];
        const end = keypoints[endIdx];
        if (!Array.isArray(start) || start.length < 3 || !Array.isArray(end) || end.length < 3) {
            console.warn(`无效的边${edgeIdx}：起点/终点格式错误`);
            return;
        }

        // 置信度判断
        const startConf = start.length >= 4 ? start[3] : 1.0;
        const endConf = end.length >= 4 ? end[3] : 1.0;
        if (startConf < 0.1 || endConf < 0.1) {
            return;
        }

        // ========== 核心修复：计算收缩后的骨骼起点/终点 ==========
        const startVec = new THREE.Vector3(start[0], start[1], start[2]);
        const endVec = new THREE.Vector3(end[0], end[1], end[2]);

        // 1. 计算关节中心到中心的向量和长度
        const direction = new THREE.Vector3().subVectors(endVec, startVec);
        const fullLength = direction.length();
        if (fullLength < 2 * radius_joint) return; // 跳过距离过近的关节（避免骨骼长度为负）

        // 2. 收缩向量：向中间收缩一个关节半径（骨骼端部贴合关节表面）
        const shrinkDistance = radius_joint; // 收缩距离 = 关节半径
        const shrinkRatio = shrinkDistance / fullLength; // 收缩比例
        const startShrink = startVec.clone().add(direction.clone().multiplyScalar(shrinkRatio)); // 起点向终点收缩
        const endShrink = endVec.clone().sub(direction.clone().multiplyScalar(shrinkRatio));     // 终点向起点收缩

        // 3. 计算收缩后的骨骼方向和长度
        const boneDirection = new THREE.Vector3().subVectors(endShrink, startShrink);
        const boneLength = boneDirection.length();

        // ========== 创建四棱锥（仅使用收缩后的参数） ==========
        const limbGeometry = new THREE.ConeGeometry(
            radius_limb,    // 底部半径
            boneLength,     // 收缩后的骨骼长度（核心：不再是完整距离）
            4,              // 四棱锥
            1,
            false
        );
        const limb = new THREE.Mesh(limbGeometry, limbMaterial);

        // ========== 调整骨骼位置和旋转（基于收缩后的点） ==========
        // 1. 位置：收缩后起点/终点的中点
        limb.position.lerpVectors(startShrink, endShrink, 0.5);
        // 2. 旋转：四棱锥Y轴对齐收缩后的骨骼方向
        const targetQuaternion = new THREE.Quaternion();
        targetQuaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0),
            boneDirection.normalize()
        );
        limb.quaternion.copy(targetQuaternion);

        // 标记为骨架元素
        limb.userData.isSkeletonElement = true;
        geometries.push(limb);
        scene.add(limb);
    });

    return geometries;
}