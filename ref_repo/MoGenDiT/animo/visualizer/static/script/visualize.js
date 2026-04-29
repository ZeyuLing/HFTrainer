/**
 * 3D骨架可视化器
 * 通用骨架结构可视化核心类
 */

class SkeletonVisualizer {
    constructor(canvasId, skeletonData) {
        this.canvas = document.getElementById(canvasId);
        this.skeletonData = skeletonData;
        this.joints = [];
        this.bones = [];
        this.jointLabels = [];
        this.animateRotation = true;
        this.rotationSpeed = 0.002;

        // 性能监控
        this.fps = 0;
        this.lastTime = 0;
        this.frameCount = 0;

        // 初始化Three.js场景
        this.initThreeJS();

        // 创建骨架
        this.createSkeleton();

        // 设置事件监听器
        this.setupEventListeners();

        // 开始动画循环
        this.animate();
    }

    /**
     * 初始化Three.js
     */
    initThreeJS() {
        // 渲染器
        this.renderer = new THREE.WebGLRenderer({
            canvas: this.canvas,
            antialias: true,
            alpha: true
        });
        this.renderer.setSize(this.canvas.clientWidth, this.canvas.clientHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.renderer.setClearColor(0x0a0a1a, 1);
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

        // 场景
        this.scene = new THREE.Scene();

        // 相机
        const aspect = this.canvas.clientWidth / this.canvas.clientHeight;
        this.camera = new THREE.PerspectiveCamera(45, aspect, 0.1, 1000);
        this.camera.position.set(5, 5, 5);

        // 轨道控制器
        this.controls = new THREE.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.dampingFactor = 0.05;
        this.controls.screenSpacePanning = true;

        // 光照
        this.setupLights();

        // 坐标系
        this.axesHelper = new THREE.AxesHelper(3);
        this.scene.add(this.axesHelper);

        // 网格
        this.gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.gridHelper.position.y = -1;
        this.scene.add(this.gridHelper);

        // 性能监控文本
        this.setupPerformanceMonitor();
    }

    /**
     * 设置光照
     */
    setupLights() {
        // 环境光
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.4);
        this.scene.add(ambientLight);

        // 平行光1 (主光源)
        const directionalLight1 = new THREE.DirectionalLight(0xffffff, 0.8);
        directionalLight1.position.set(5, 10, 5);
        directionalLight1.castShadow = true;
        directionalLight1.shadow.camera.left = -10;
        directionalLight1.shadow.camera.right = 10;
        directionalLight1.shadow.camera.top = 10;
        directionalLight1.shadow.camera.bottom = -10;
        this.scene.add(directionalLight1);

        // 平行光2 (补光)
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.4);
        directionalLight2.position.set(-5, 3, -5);
        this.scene.add(directionalLight2);

        // 点光源
        const pointLight = new THREE.PointLight(0x4dabf7, 0.6, 20);
        pointLight.position.set(0, 3, 0);
        this.scene.add(pointLight);
    }

    /**
     * 设置性能监控
     */
    setupPerformanceMonitor() {
        const fpsElement = document.getElementById('fps');
        if (fpsElement) {
            this.fpsElement = fpsElement;
        }
    }

    /**
     * 创建骨架
     */
    createSkeleton() {
        this.clearSkeleton();

        if (!this.skeletonData) {
            console.error("没有提供骨架数据");
            return;
        }

        const jointOffsets = this.skeletonData.jointOffsets || [];
        const kinematicTree = this.skeletonData.kinematicTree || {};
        const boneDensity = this.skeletonData.boneDensity || {};
        const jointNames = this.skeletonData.jointNames || [];

        // 计算关节的世界坐标
        const jointPositions = this.calculateJointPositions(jointOffsets, kinematicTree);

        // 创建关节
        this.createJoints(jointPositions, jointNames);

        // 创建骨骼
        this.createBones(jointPositions, kinematicTree, boneDensity);

        // 创建关节标签
        this.createJointLabels(jointPositions, jointNames);

        // 更新UI信息
        this.updateJointInfo(jointPositions, jointNames);
        this.updateSkeletonInfo();
    }

    /**
     * 计算关节位置
     */
    calculateJointPositions(jointOffsets, kinematicTree) {
        const positions = [];

        // 初始化所有关节位置
        for (let i = 0; i < jointOffsets.length; i++) {
            positions.push(new THREE.Vector3(0, 0, 0));
        }

        if (jointOffsets.length > 0) {
            positions[0].copy(jointOffsets[0]);
        }

        // 使用广度优先搜索遍历运动树
        const queue = [0];
        const visited = new Set([0]);

        while (queue.length > 0) {
            const current = queue.shift();

            for (const [level, edges] of Object.entries(kinematicTree)) {
                for (const [parent, child] of edges) {
                    if (parent === current && !visited.has(child)) {
                        positions[child].copy(positions[parent]).add(jointOffsets[child]);
                        visited.add(child);
                        queue.push(child);
                    }
                }
            }
        }

        return positions;
    }

    /**
     * 创建关节
     */
    createJoints(jointPositions, jointNames) {
        const jointSize = parseFloat(document.getElementById('jointSize').value) || 0.5;

        for (let i = 0; i < jointPositions.length; i++) {
            const position = jointPositions[i];
            const jointName = jointNames[i] || `关节 ${i}`;

            // 根据关节类型选择颜色
            let color;
            if (i === 0) {
                color = 0xff0000; // 根节点 - 红色
            } else if (jointName.includes('left')) {
                color = 0x00ff00; // 左侧关节 - 绿色
            } else if (jointName.includes('right')) {
                color = 0x0000ff; // 右侧关节 - 蓝色
            } else if (i % 2 === 0) {
                color = 0xffaa00; // 偶数节点 - 橙色
            } else {
                color = 0xff00ff; // 奇数节点 - 紫色
            }

            // 创建关节球体
            const geometry = new THREE.SphereGeometry(jointSize, 16, 16);
            const material = new THREE.MeshPhongMaterial({
                color: color,
                shininess: 100,
                transparent: true,
                opacity: 0.9
            });

            const joint = new THREE.Mesh(geometry, material);
            joint.position.copy(position);
            joint.userData = {
                jointId: i,
                jointName: jointName,
                isJoint: true
            };

            this.scene.add(joint);
            this.joints.push(joint);
        }
    }

    /**
     * 创建骨骼
     */
    createBones(jointPositions, kinematicTree, boneDensity) {
        const boneThickness = parseFloat(document.getElementById('boneThickness').value) || 0.3;

        // 从运动树中提取所有骨骼连接
        const boneEdges = [];
        for (const [level, edges] of Object.entries(kinematicTree)) {
            for (const [parent, child] of edges) {
                boneEdges.push([parent, child]);
            }
        }

        for (const [parentIdx, childIdx] of boneEdges) {
            const parentPos = jointPositions[parentIdx];
            const childPos = jointPositions[childIdx];

            const direction = new THREE.Vector3().subVectors(childPos, parentPos);
            const length = direction.length();

            if (length === 0) continue;

            direction.normalize();

            // 创建骨骼圆柱体
            const geometry = new THREE.CylinderGeometry(
                boneThickness,
                boneThickness * 0.9, // 稍微锥形
                length,
                8
            );
            geometry.rotateZ(Math.PI / 2);

            // 根据骨骼密度选择颜色
            const densityKey = `(${parentIdx}, ${childIdx})`;
            const density = boneDensity[densityKey] || boneDensity[`${parentIdx},${childIdx}`] || 0.5;

            let boneColor;
            if (density > 0.7) {
                boneColor = 0xffaa00; // 高密度 - 橙色
            } else if (density > 0.3) {
                boneColor = 0xffff00; // 中密度 - 黄色
            } else {
                boneColor = 0x00aaff; // 低密度 - 蓝色
            }

            const material = new THREE.MeshPhongMaterial({
                color: boneColor,
                shininess: 30,
                transparent: true,
                opacity: 0.8
            });

            const bone = new THREE.Mesh(geometry, material);

            // 将骨骼放置在父节点和子节点中间
            const center = new THREE.Vector3().addVectors(parentPos, childPos).multiplyScalar(0.5);
            bone.position.copy(center);

            // 设置骨骼方向
            bone.lookAt(childPos);
            bone.rotateX(Math.PI / 2);

            bone.userData = {
                parentId: parentIdx,
                childId: childIdx,
                isBone: true
            };

            this.scene.add(bone);
            this.bones.push(bone);
        }
    }

    /**
     * 创建关节标签
     */
    createJointLabels(jointPositions, jointNames) {
        for (let i = 0; i < jointPositions.length; i++) {
            const position = jointPositions[i];
            const jointName = jointNames[i] || `J${i}`;

            // 创建Canvas用于文本渲染
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 128;
            canvas.height = 48;

            // 绘制文本背景
            context.fillStyle = 'rgba(0, 0, 0, 0.7)';
            context.fillRect(0, 0, canvas.width, canvas.height);

            // 绘制边框
            context.strokeStyle = '#4dabf7';
            context.lineWidth = 2;
            context.strokeRect(1, 1, canvas.width - 2, canvas.height - 2);

            // 绘制文本
            context.font = 'bold 14px Arial';
            context.fillStyle = '#ffffff';
            context.textAlign = 'center';
            context.textBaseline = 'middle';
            context.fillText(jointName, canvas.width / 2, canvas.height / 2 - 8);

            context.font = '12px Arial';
            context.fillStyle = '#cccccc';
            context.fillText(`ID: ${i}`, canvas.width / 2, canvas.height / 2 + 8);

            // 创建纹理
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({
                map: texture,
                transparent: true
            });
            const sprite = new THREE.Sprite(spriteMaterial);

            // 设置精灵位置（在关节上方）
            sprite.position.copy(position);
            sprite.position.y += 0.5;
            sprite.scale.set(2, 1, 1);

            sprite.userData = {
                jointId: i,
                jointName: jointName,
                isLabel: true
            };

            this.scene.add(sprite);
            this.jointLabels.push(sprite);
        }
    }

    /**
     * 更新关节信息显示
     */
    updateJointInfo(jointPositions, jointNames) {
        const jointInfoDiv = document.getElementById('jointInfo');
        if (!jointInfoDiv) return;

        jointInfoDiv.innerHTML = '';

        for (let i = 0; i < jointPositions.length; i++) {
            const pos = jointPositions[i];
            const jointName = jointNames[i] || `关节 ${i}`;

            const jointItem = document.createElement('div');
            jointItem.className = 'joint-item';
            jointItem.innerHTML = `
                <div>
                    <span class="joint-id">${jointName}</span>
                    <div style="font-size: 0.8rem; color: #888; margin-top: 2px;">ID: ${i}</div>
                </div>
                <span class="joint-coords">(${pos.x.toFixed(2)}, ${pos.y.toFixed(2)}, ${pos.z.toFixed(2)})</span>
            `;

            jointItem.addEventListener('click', () => {
                // 移除之前选中的样式
                document.querySelectorAll('.joint-item.selected').forEach(el => {
                    el.classList.remove('selected');
                });

                // 添加当前选中的样式
                jointItem.classList.add('selected');

                // 聚焦到关节
                this.focusOnJoint(i);
            });

            jointInfoDiv.appendChild(jointItem);
        }
    }

    /**
     * 更新骨架信息显示
     */
    updateSkeletonInfo() {
        if (!this.skeletonData) return;

        const skeletonName = document.getElementById('currentSkeleton');
        const jointCount = document.getElementById('jointCount');
        const boneCount = document.getElementById('boneCount');

        if (skeletonName) {
            skeletonName.textContent = this.skeletonData.name || "未知骨架";
        }

        if (jointCount) {
            jointCount.textContent = this.joints.length;
        }

        if (boneCount) {
            boneCount.textContent = this.bones.length;
        }
    }

    /**
     * 更新FPS显示
     */
    updateFPS() {
        if (!this.fpsElement) return;

        const now = performance.now();
        this.frameCount++;

        if (now >= this.lastTime + 1000) {
            this.fps = Math.round((this.frameCount * 1000) / (now - this.lastTime));
            this.fpsElement.textContent = `${this.fps} FPS`;
            this.frameCount = 0;
            this.lastTime = now;
        }
    }

    /**
     * 聚焦到指定关节
     */
    focusOnJoint(jointId) {
        if (jointId >= 0 && jointId < this.joints.length) {
            const joint = this.joints[jointId];
            this.controls.target.copy(joint.position);

            // 计算相机位置，使其朝向关节
            const distance = 3;
            const cameraPosition = joint.position.clone();
            cameraPosition.x += distance;
            cameraPosition.y += distance;
            cameraPosition.z += distance;

            this.camera.position.copy(cameraPosition);
            this.controls.update();
        }
    }

    /**
     * 清理骨架
     */
    clearSkeleton() {
        // 移除所有关节
        for (const joint of this.joints) {
            this.scene.remove(joint);
            if (joint.geometry) joint.geometry.dispose();
            if (joint.material) joint.material.dispose();
        }
        this.joints = [];

        // 移除所有骨骼
        for (const bone of this.bones) {
            this.scene.remove(bone);
            if (bone.geometry) bone.geometry.dispose();
            if (bone.material) bone.material.dispose();
        }
        this.bones = [];

        // 移除所有标签
        for (const label of this.jointLabels) {
            this.scene.remove(label);
            if (label.material.map) label.material.map.dispose();
            if (label.material) label.material.dispose();
        }
        this.jointLabels = [];
    }

    /**
     * 设置事件监听器
     */
    setupEventListeners() {
        // 窗口大小调整
        window.addEventListener('resize', () => {
            this.onWindowResize();
        });

        // 显示/隐藏控制
        document.getElementById('showJoints').addEventListener('change', (e) => {
            this.toggleJointsVisibility(e.target.checked);
        });

        document.getElementById('showBones').addEventListener('change', (e) => {
            this.toggleBonesVisibility(e.target.checked);
        });

        document.getElementById('showLabels').addEventListener('change', (e) => {
            this.toggleLabelsVisibility(e.target.checked);
        });

        document.getElementById('showAxes').addEventListener('change', (e) => {
            this.axesHelper.visible = e.target.checked;
        });

        document.getElementById('showGrid').addEventListener('change', (e) => {
            this.gridHelper.visible = e.target.checked;
        });

        // 滑块控制
        document.getElementById('jointSize').addEventListener('input', (e) => {
            document.getElementById('jointSizeValue').textContent = e.target.value;
            this.updateJointSize(parseFloat(e.target.value));
        });

        document.getElementById('boneThickness').addEventListener('input', (e) => {
            document.getElementById('boneThicknessValue').textContent = e.target.value;
            this.updateBoneThickness(parseFloat(e.target.value));
        });

        document.getElementById('rotationSpeed').addEventListener('input', (e) => {
            document.getElementById('rotationSpeedValue').textContent = e.target.value;
            this.rotationSpeed = parseFloat(e.target.value) * 0.004;
        });

        // 关节点击事件
        this.canvas.addEventListener('dblclick', (event) => {
            const rect = this.canvas.getBoundingClientRect();
            const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            const raycaster = new THREE.Raycaster();
            raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);

            const intersects = raycaster.intersectObjects([...this.joints, ...this.bones]);
            if (intersects.length > 0) {
                const object = intersects[0].object;
                if (object.userData.isJoint) {
                    this.focusOnJoint(object.userData.jointId);
                } else if (object.userData.isBone) {
                    this.focusOnJoint(object.userData.childId);
                }
            }
        });
    }

    /**
     * 切换关节可见性
     */
    toggleJointsVisibility(visible) {
        for (const joint of this.joints) {
            joint.visible = visible;
        }
    }

    /**
     * 切换骨骼可见性
     */
    toggleBonesVisibility(visible) {
        for (const bone of this.bones) {
            bone.visible = visible;
        }
    }

    /**
     * 切换标签可见性
     */
    toggleLabelsVisibility(visible) {
        for (const label of this.jointLabels) {
            label.visible = visible;
        }
    }

    /**
     * 更新关节大小
     */
    updateJointSize(size) {
        for (let i = 0; i < this.joints.length; i++) {
            const joint = this.joints[i];
            const oldGeometry = joint.geometry;

            const newGeometry = new THREE.SphereGeometry(size, 16, 16);
            joint.geometry = newGeometry;

            if (oldGeometry) oldGeometry.dispose();
        }
    }

    /**
     * 更新骨骼粗细
     */
    updateBoneThickness(thickness) {
        for (let i = 0; i < this.bones.length; i++) {
            const bone = this.bones[i];
            const oldGeometry = bone.geometry;

            const length = bone.geometry.parameters.height;
            const newGeometry = new THREE.CylinderGeometry(
                thickness,
                thickness * 0.9,
                length,
                8
            );
            newGeometry.rotateZ(Math.PI / 2);
            bone.geometry = newGeometry;

            if (oldGeometry) oldGeometry.dispose();
        }
    }

    /**
     * 重置视图
     */
    resetView() {
        this.controls.reset();
        this.camera.position.set(5, 5, 5);
        this.controls.target.set(0, 0, 0);
        this.controls.update();
    }

    /**
     * 导出截图
     */
    exportScreenshot() {
        const link = document.createElement('a');
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        link.download = `skeleton_${timestamp}.png`;

        this.renderer.render(this.scene, this.camera);
        link.href = this.renderer.domElement.toDataURL('image/png');
        link.click();

        // 显示成功消息
        const originalText = document.getElementById('exportView').textContent;
        document.getElementById('exportView').textContent = '已保存!';
        setTimeout(() => {
            document.getElementById('exportView').textContent = originalText;
        }, 1500);
    }

    /**
     * 窗口大小调整
     */
    onWindowResize() {
        const width = this.canvas.clientWidth;
        const height = this.canvas.clientHeight;

        this.camera.aspect = width / height;
        this.camera.updateProjectionMatrix();

        this.renderer.setSize(width, height, false);
    }

    /**
     * 动画循环
     */
    animate() {
        requestAnimationFrame(() => this.animate());

        // 自动旋转
        if (this.animateRotation) {
            this.scene.rotation.y += this.rotationSpeed;
        }

        this.controls.update();
        this.renderer.render(this.scene, this.camera);

        // 更新FPS
        this.updateFPS();
    }

    /**
     * 更新骨架数据
     */
    updateSkeletonData(newSkeletonData) {
        this.skeletonData = newSkeletonData;
        this.createSkeleton();
    }

    /**
     * 切换自动旋转
     */
    toggleRotation() {
        this.animateRotation = !this.animateRotation;
        return this.animateRotation;
    }
}

// 导出全局类
window.SkeletonVisualizer = SkeletonVisualizer;
