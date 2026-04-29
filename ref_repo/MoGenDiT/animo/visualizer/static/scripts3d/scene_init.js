
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { create_scene } from '/static/scripts3d/create_scene.js';
// import { load_smpl_with_shapes } from '/static/scripts3d/load_smpl_baseline.js';

// 颜色配置
const DEFAULT_COLORS = [
    0xffffff,  // 0 - 白色 (中性)
    0x6495ED,  // 1 - 蓝色 (算法1/基线1)
    0xFF6B81,  // 2 - 红色 (算法2/基线2) 
    0x32CD32,  // 3 - 绿色 (算法3/基线3)
    0xFFD700,  // 4 - 金色 (算法4/基线4)
    0xFF69B4,  // 5 - 粉色 (算法5/基线5)
    0x888888,  // 6 - 灰色 (算法6/基线6)
    0x4CAF50,  // 7 - 深绿色 (算法7/基线7)
    0xFF8C00,  // 8 - 橙色 (算法8/基线8)
    0x9370DB   // 9 - 紫色 (算法9/基线9)
];

// 颜色名称映射
const COLOR_NAMES = [
    "白色 (方法0)",     // 0
    "蓝色 (方法1)",    // 1  
    "红色 (方法2)",    // 2
    "绿色 (方法3)",    // 3
    "金色 (方法4)",    // 4
    "粉色 (方法5)",    // 5
    "灰色 (方法6)",    // 6
    "深绿 (方法7)",    // 7
    "橙色 (方法8)",    // 8
    "紫色 (方法9)"     // 9
];

// 全局3D场景变量
let scene, camera, renderer;
let controls;
let infos;
let currentFrame = 0;
let total_frame = 0;
let totalDuration = 0; // 总播放时长（秒）
const intervalTime = 30.; // 每帧之间的时间间隔，单位毫秒
var model_mesh = {};
let intervalId = null; // 存储播放间隔ID
let isPlaying = true; // 播放状态
let playbackSpeed = 1; // 播放速度

// 新增：相机相关变量
let perspectiveCamera; // 透视相机
let orthographicCamera; // 正交相机
let isOrthographicMode = false; // 当前是否为正交模式
const CAMERA_DISTANCE = 5; // 相机距离目标的距离
const ORTHO_BASE_SIZE = 5; // 正交相机基础视口大小
const MAX_RENDER_DISTANCE = 1000; // 增大渲染距离

// 控制器基础配置
const CONTROLS_CONFIG = {
    perspective: { minDistance: 0.5, maxDistance: 50, enableZoom: true },
    orthographic: { minDistance: 0.5, maxDistance: 1000, enableZoom: false }
};

// 正交相机缩放参数（可调灵敏度）
const ORTHO_ZOOM_CONFIG = {
    speed: 0.05, // 缩放步长
    minZoom: 0.05,
    maxZoom: 30
};

const SMPL_TPOSE_JOINTS = [
    [0.000000, 0.000000, 0.000000],       // 0: 根关节（骨盆）
    [0.058581, -0.082280, -0.017664],     // 1: 右髋
    [-0.060310, -0.090513, -0.013543],    // 2: 左髋
    [0.004439, 0.124404, -0.038385],      // 3: 脊柱1
    [0.102033, -0.468750, -0.009627],     // 4: 右膝
    [-0.103566, -0.474201, -0.018386],    // 5: 左膝
    [0.008928, 0.262360, -0.011565],      // 6: 脊柱2
    [0.087242, -0.895624, -0.047055],     // 7: 右踝
    [-0.084511, -0.894247, -0.052947],    // 8: 左踝
    [0.006663, 0.318392, -0.008710],      // 9: 脊柱3
    [0.128297, -0.955910, 0.074987],      // 10: 右脚
    [-0.119351, -0.956352, 0.077376],     // 11: 左脚
    [-0.006727, 0.530028, -0.042178],     // 12: 颈部
    [0.078366, 0.432392, -0.027608],      // 13: 右肩
    [-0.076291, 0.430865, -0.032417],     // 14: 左肩
    [0.003386, 0.618965, 0.008232],       // 15: 头部
    [0.201287, 0.477597, -0.046654],      // 16: 右肘
    [-0.189519, 0.477718, -0.040889],     // 17: 左肘
    [0.456619, 0.461948, -0.069601],      // 18: 右腕
    [-0.449646, 0.463349, -0.072158],     // 19: 左腕
    [0.722328, 0.474646, -0.076975],      // 20: 右手
    [-0.718755, 0.471286, -0.078185],     // 21: 左手
    [0.809019, 0.464010, -0.092570],      // 22: 右手指
    [-0.807508, 0.444770, -0.088292]      // 23: 左手指
];

const edges = [[0, 1], [0, 2], [0, 3],
[1, 4], [2, 5], [3, 6],
[4, 7], [5, 8], [6, 9],
[7, 10], [8, 11], [9, 12], [9, 13], [9, 14],
[12, 15], [13, 16], [14, 17],
[16, 18], [17, 19],
[18, 20], [19, 21],
[20, 22], [21, 23]
];

// 初始化3D场景
function init() {
    const container = document.getElementById('vis3d');
    const canvas = document.getElementById('scene-canvas');

    // 兜底：确保关键元素存在
    if (!canvas || !container) {
        console.error('缺少必要元素：canvas或vis3d容器');
        return;
    }

    const width = container.offsetWidth || 1200;
    const height = width * 9 / 16;

    scene = new THREE.Scene();

    // 1. 透视相机初始化
    perspectiveCamera = new THREE.PerspectiveCamera(
        60, width / height, 0.1, MAX_RENDER_DISTANCE
    );
    perspectiveCamera.position.set(CAMERA_DISTANCE, CAMERA_DISTANCE, CAMERA_DISTANCE);
    perspectiveCamera.lookAt(0, 1, 0);

    // 2. 正交相机初始化（核心：结合zoom计算视口，避免裁剪）
    orthographicCamera = createOrthographicCamera(width, height);
    orthographicCamera.position.copy(perspectiveCamera.position);
    orthographicCamera.lookAt(0, 1, 0);

    camera = perspectiveCamera;

    // 渲染器初始化
    renderer = new THREE.WebGLRenderer({
        canvas: canvas, antialias: true, logarithmicDepthBuffer: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);

    // 场景基础设置（确保create_scene存在）
    if (typeof create_scene === 'function') {
        create_scene(scene, camera, renderer, true, 'y');
    }

    // 初始化控制器
    initControls();

    // 滚轮事件绑定（全局+兼容）
    document.addEventListener('wheel', onMouseWheel, {
        passive: false, capture: true
    });

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

    drawSkeleton(SMPL_TPOSE_JOINTS, edges, scene);

    // 动画循环
    function animate() {
        requestAnimationFrame(animate);
        controls.update();
        renderer.render(scene, camera);
    }
    animate();

    // 窗口大小变化监听（核心：同步更新正交相机视口）
    window.addEventListener('resize', () => onWindowResize(width, height));

    // console.log('初始化完成，当前相机：', isOrthographicMode ? '正交' : '透视');
}

// ========== 核心函数：创建/更新正交相机（解决底部裁剪） ==========
function createOrthographicCamera(width, height) {
    const aspect = width / height;
    // 基础视口大小 = 基础尺寸 * 缩放比例（zoom越小，视口越大，避免裁剪）
    const baseSize = ORTHO_BASE_SIZE / (orthographicCamera?.zoom || 1);

    const orthoCam = new THREE.OrthographicCamera(
        -baseSize * aspect,  // left
        baseSize * aspect,   // right
        baseSize,            // top
        -baseSize,           // bottom（对称设置，彻底解决底部裁剪）
        0.1,                 // near（必须小，避免近裁剪）
        MAX_RENDER_DISTANCE  // far
    );
    // 初始化zoom（首次创建）
    if (!orthographicCamera) orthoCam.zoom = 1;

    orthoCam.updateProjectionMatrix();
    return orthoCam;
}

// 初始化/重置控制器
function initControls() {
    if (controls) controls.dispose();

    controls = new OrbitControls(camera, renderer.domElement);
    const config = isOrthographicMode ? CONTROLS_CONFIG.orthographic : CONTROLS_CONFIG.perspective;

    controls.minDistance = config.minDistance;
    controls.maxDistance = config.maxDistance;
    controls.enableZoom = config.enableZoom;
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 1, 0); // 固定目标点，避免偏移
    controls.update();
}

// ========== 修复滚轮+同步更新正交相机视口（解决裁剪） ==========
function onMouseWheel(event) {
    if (!isOrthographicMode) return;

    event.preventDefault(); // 阻止页面滚动

    // 兼容滚轮方向
    const delta = event.deltaY || event.deltaX;
    if (!delta) return;

    // 计算新zoom
    let newZoom = orthographicCamera.zoom;
    newZoom += delta > 0 ? -ORTHO_ZOOM_CONFIG.speed : ORTHO_ZOOM_CONFIG.speed;
    newZoom = Math.max(ORTHO_ZOOM_CONFIG.minZoom, Math.min(newZoom, ORTHO_ZOOM_CONFIG.maxZoom));

    // 关键：更新zoom后，重新计算正交相机视口（避免裁剪）
    orthographicCamera.zoom = newZoom;
    const container = document.getElementById('vis3d');
    const width = container.offsetWidth || 1200;
    const height = width * 9 / 16;
    // 重新赋值left/right/top/bottom
    const aspect = width / height;
    const baseSize = ORTHO_BASE_SIZE / newZoom;
    orthographicCamera.left = -baseSize * aspect;
    orthographicCamera.right = baseSize * aspect;
    orthographicCamera.top = baseSize;
    orthographicCamera.bottom = -baseSize; // 对称，无裁剪

    orthographicCamera.updateProjectionMatrix();
    console.log('正交相机参数：', {
        zoom: orthographicCamera.zoom,
        bottom: orthographicCamera.bottom,
        top: orthographicCamera.top
    });
}

// 窗口大小变化处理
function onWindowResize() {
    const container = document.getElementById('vis3d');
    const width = container.offsetWidth || 1200;
    const height = width * 9 / 16;

    renderer.setSize(width, height);

    // 更新透视相机
    perspectiveCamera.aspect = width / height;
    perspectiveCamera.updateProjectionMatrix();

    // 更新正交相机（核心：同步视口，避免窗口缩放后裁剪）
    if (isOrthographicMode) {
        orthographicCamera = createOrthographicCamera(width, height);
        camera = orthographicCamera; // 同步camera引用
    }
}

// 切换相机模式（同步更新正交相机视口）
window.toggleCameraMode = function () {
    const container = document.getElementById('vis3d');
    const cameraSwitchBtn = document.getElementById('cameraSwitchBtn');
    if (!container || !cameraSwitchBtn) return;

    isOrthographicMode = !isOrthographicMode;
    const width = container.offsetWidth;
    const height = container.offsetHeight || width * 9 / 16;

    // 保存当前位置
    const position = camera.position.clone();
    const target = controls.target.clone();

    if (isOrthographicMode) {
        // 切换到正交：先更新视口，再赋值
        orthographicCamera = createOrthographicCamera(width, height);
        camera = orthographicCamera;
        cameraSwitchBtn.innerHTML = '<i class="fas fa-cubes"></i>';
        cameraSwitchBtn.title = '切换到透视视角';
    } else {
        // 切换到透视
        camera = perspectiveCamera;
        perspectiveCamera.aspect = width / height;
        perspectiveCamera.updateProjectionMatrix();
        cameraSwitchBtn.innerHTML = '<i class="fas fa-cube"></i>';
        cameraSwitchBtn.title = '切换到正交视角';
    }

    // 恢复位置和目标
    camera.position.copy(position);
    camera.lookAt(target);

    // 重新初始化控制器
    initControls();
    controls.target.copy(target);
    controls.update();

    console.log('切换到', isOrthographicMode ? '正交' : '透视', '模式');
};


// 页面加载完成初始化
document.addEventListener('DOMContentLoaded', function () {
    init();
});


// 1. 启动动捕监听
async function startMocap() {
    const res = await fetch("/mocap/start");
    const data = await res.json();
    console.log("启动结果：", data);
    if (data.code === 200) {
        alert(data.msg);
    }
}

// 2. 停止动捕监听
async function stopMocap() {
    const res = await fetch("/mocap/stop");
    const data = await res.json();
    console.log("停止结果：", data);
    if (data.code === 200) {
        alert(data.msg);
    }
}

// 3. 获取当前状态
async function getMocapStatus() {
    const res = await fetch("/mocap/status");
    const data = await res.json();
    console.log("当前状态：", data);
    return data.data;
}



