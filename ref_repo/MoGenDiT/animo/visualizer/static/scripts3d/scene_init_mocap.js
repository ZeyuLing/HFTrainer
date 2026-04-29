import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { create_scene } from '/static/scripts3d/create_scene.js';

// 颜色配置/常量定义（保留不变）
const DEFAULT_COLORS = [0xffffff, 0x6495ED, 0xFF6B81, 0x32CD32, 0xFFD700, 0xFF69B4, 0x888888, 0x4CAF50, 0xFF8C00, 0x9370DB];
const COLOR_NAMES = ["白色 (方法0)", "蓝色 (方法1)", "红色 (方法2)", "绿色 (方法3)", "金色 (方法4)", "粉色 (方法5)", "灰色 (方法6)", "深绿 (方法7)", "橙色 (方法8)", "紫色 (方法9)"];

// 全局变量（新增：节流相关+复用对象）
let scene, camera, renderer;
let controls;
let currentFrame = 0;
let total_frame = 0;
let totalDuration = 0;
const intervalTime = 30.;
var model_mesh = {};
let intervalId = null;
let isPlaying = true;
let playbackSpeed = 1;

// 相机相关（保留）
let perspectiveCamera;
let orthographicCamera;
let isOrthographicMode = false;
const CAMERA_DISTANCE = 5;
const ORTHO_BASE_SIZE = 5;
const MAX_RENDER_DISTANCE = 1000;

const CONTROLS_CONFIG = {
    perspective: { minDistance: 0.5, maxDistance: 50, enableZoom: true },
    orthographic: { minDistance: 0.5, maxDistance: 1000, enableZoom: false }
};

const ORTHO_ZOOM_CONFIG = {
    speed: 0.05,
    minZoom: 0.05,
    maxZoom: 30
};
const body_h = 0.956352
const SMPL_TPOSE_JOINTS_RAW = [
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
const SMPL_TPOSE_JOINTS = [];
for (let i = 0; i < SMPL_TPOSE_JOINTS_RAW.length; i++) {
    // 复制原始关节的x、z轴，Y轴 = 原始值 + body_h
    const [x, y, z] = SMPL_TPOSE_JOINTS_RAW[i];
    SMPL_TPOSE_JOINTS.push([x, y + body_h, z]);
}

const edges = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]];

// 新增：复用Vector3/Quaternion（减少GC）
const tempVec3_1 = new THREE.Vector3();
const tempVec3_2 = new THREE.Vector3();
const tempVec3_3 = new THREE.Vector3();
const tempQuaternion = new THREE.Quaternion();
const upVec = new THREE.Vector3(0, 1, 0);

// 全局骨架变量（保留）
let jointMeshes = [];
let boneMeshes = [];
let socket;
let isMocapActive = false;
const RADIUS_JOINT = 0.015;
const RADIUS_LIMB = 0.015;

// 新增：节流控制（核心优化）
let lastUpdateTime = 0;
const UPDATE_INTERVAL = 16; // 约60fps，控制最大更新频率
let pendingJoints = null; // 待处理的关节数据（节流用）

// 初始化3D场景（保留，仅修复resize绑定）
function init() {
    const container = document.getElementById('vis3d');
    const canvas = document.getElementById('scene-canvas');

    if (!canvas || !container) {
        console.error('缺少必要元素：canvas或vis3d容器');
        return;
    }

    const width = container.offsetWidth || 1200;
    const height = width * 9 / 16;

    scene = new THREE.Scene();
    // 新增：添加基础光源（解决MeshStandardMaterial看不见的问题）
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.8);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.6);
    directionalLight.position.set(5, 5, 5);
    scene.add(ambientLight);
    scene.add(directionalLight);

    // 相机初始化（保留，微调初始位置）
    perspectiveCamera = new THREE.PerspectiveCamera(60, width / height, 0.1, MAX_RENDER_DISTANCE);
    perspectiveCamera.position.set(2, 2, 2); // 从5,5,5缩小到2,2,2，更靠近骨架
    perspectiveCamera.lookAt(0, 0, 0); // 看向根关节(0,0,0)，而非(0,1,0)

    orthographicCamera = createOrthographicCamera(width, height);
    orthographicCamera.position.copy(perspectiveCamera.position);
    orthographicCamera.lookAt(0, 0, 0); // 同步看向根关节

    camera = perspectiveCamera;

    // 渲染器（保留）
    renderer = new THREE.WebGLRenderer({
        canvas: canvas, antialias: true, logarithmicDepthBuffer: true
    });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(width, height);
    // 新增：设置场景背景色（避免和骨架颜色冲突）
    scene.background = new THREE.Color(0xf9f9f9); // 浅灰色背景

    if (typeof create_scene === 'function') {
        create_scene(scene, camera, renderer, true, 'y');
    }

    initControls();
    document.addEventListener('wheel', onMouseWheel, { passive: false, capture: true });

    // 初始化骨架（强制执行，不受节流影响）
    initSkeleton();
    updateSkeleton(SMPL_TPOSE_JOINTS);
    // 新增：强制渲染一次，确保初始骨架显示
    renderer.render(scene, camera);

    // 初始化WebSocket
    initWebSocket();

    // 动画循环（修复节流逻辑）
    function animate() {
        requestAnimationFrame(animate);
        controls.update();

        // 修复：节流仅作用于动捕数据，初始骨架不受影响
        if (pendingJoints) {
            const now = performance.now();
            if (now - lastUpdateTime > UPDATE_INTERVAL) {
                updateSkeleton(pendingJoints);
                pendingJoints = null;
                lastUpdateTime = now;
            }
        }
        // 移除else判断，确保无论是否有pendingJoints，都持续渲染
        renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', onWindowResize);

    console.log('初始化完成，当前相机：', isOrthographicMode ? '正交' : '透视');
}

// ========== 核心优化1：骨架初始化（保留，无修改） ==========
function initSkeleton() {
    scene.children.forEach(child => {
        if (child.userData.isSkeletonElement) {
            scene.remove(child);
            if (child.geometry) child.geometry.dispose();
            if (child.material) child.material.dispose();
        }
    });

    jointMeshes = [];
    boneMeshes = [];

    const jointMaterial = new THREE.MeshStandardMaterial({ color: 0x6600ff });
    const limbMaterial = new THREE.MeshStandardMaterial({ color: 0x9999ff });

    // 创建关节（保留）
    for (let i = 0; i < 24; i++) {
        const sphereGeometry = new THREE.SphereGeometry(RADIUS_JOINT, 16, 16);
        const sphere = new THREE.Mesh(sphereGeometry, jointMaterial);
        sphere.userData.isSkeletonElement = true;
        jointMeshes.push(sphere);
        scene.add(sphere);
    }

    // 创建骨骼（保留）
    edges.forEach((edge, idx) => {
        const limbGeometry = new THREE.ConeGeometry(RADIUS_LIMB, 0.1, 4, 1);
        const limb = new THREE.Mesh(limbGeometry, limbMaterial);
        limb.userData.isSkeletonElement = true;
        boneMeshes.push({
            mesh: limb,
            edge: edge
        });
        scene.add(limb);
    });
}

// ========== 核心优化2：更新骨架（移除几何体重建，改用缩放） ==========
function updateSkeleton(keypoints) {
    if (!Array.isArray(keypoints) || keypoints.length !== 24) {
        console.warn('无效的关节数据：数量不为24');
        return;
    }

    // 1. 更新关节位置（保留）
    keypoints.forEach((point, idx) => {
        if (!Array.isArray(point) || point.length < 3 || !jointMeshes[idx]) return;
        const [x, y, z] = point;
        jointMeshes[idx].position.set(x, y, z);
    });

    // 2. 更新骨骼（核心优化：不重建几何体，改用scale.y缩放）
    boneMeshes.forEach((bone, idx) => {
        const [startIdx, endIdx] = bone.edge;
        const start = keypoints[startIdx];
        const end = keypoints[endIdx];

        if (!Array.isArray(start) || start.length < 3 || !Array.isArray(end) || end.length < 3) {
            bone.mesh.visible = false;
            return;
        }

        // 复用临时变量（减少GC）
        tempVec3_1.set(start[0], start[1], start[2]); // startVec
        tempVec3_2.set(end[0], end[1], end[2]);       // endVec

        // 计算方向和长度
        tempVec3_3.subVectors(tempVec3_2, tempVec3_1); // direction
        const fullLength = tempVec3_3.length();

        if (fullLength < 2 * RADIUS_JOINT) {
            bone.mesh.visible = false;
            return;
        }
        bone.mesh.visible = true;

        // 收缩骨骼（复用变量）
        const shrinkRatio = RADIUS_JOINT / fullLength;
        tempVec3_1.add(tempVec3_3.clone().multiplyScalar(shrinkRatio)); // startShrink
        tempVec3_2.sub(tempVec3_3.clone().multiplyScalar(shrinkRatio)); // endShrink
        tempVec3_3.subVectors(tempVec3_2, tempVec3_1); // boneDirection
        const boneLength = tempVec3_3.length();

        // 核心优化：用scale.y代替重建几何体
        bone.mesh.scale.y = boneLength / 0.1; // 0.1是初始几何体长度
        // 保留位置和旋转更新
        bone.mesh.position.lerpVectors(tempVec3_1, tempVec3_2, 0.5);
        tempQuaternion.setFromUnitVectors(upVec, tempVec3_3.normalize());
        bone.mesh.quaternion.copy(tempQuaternion);
    });
}

// ========== 核心优化3：WebSocket连接（修复重连+节流） ==========
function initWebSocket() {
    // 修复1：先关闭旧socket（避免多实例）
    if (socket) {
        socket.disconnect();
        socket = null;
    }

    // 连接服务端（保留）
    socket = io('http://' + window.location.hostname + ':5000', {
        reconnection: true, // 启用内置重连（替代手动重连）
        reconnectionAttempts: 10,
        reconnectionDelay: 3000
    });

    // 连接成功（保留，移除重复请求）
    socket.on('connect', () => {
        console.log('✅ WebSocket连接成功，等待动捕数据...');
        // 仅在动捕激活时请求最新数据
        if (isMocapActive) {
            socket.emit('request_latest_data');
        }
    });

    // 接收数据（核心优化：节流处理）
    socket.on('mocap_update', (data) => {
        const joints = data.joints;
        if (joints) {
            // 不立即更新，存入待处理队列（避免阻塞）
            pendingJoints = joints;
            // 仅调试用，减少日志输出
            // console.log('📤 接收动捕数据，加入更新队列');
        }
    });

    // 修复2：移除手动重连（用SocketIO内置重连）
    socket.on('disconnect', (reason) => {
        console.log(`🔌 WebSocket断开：${reason}，内置重连将在3秒后尝试`);
    });

    socket.on('connect_error', (error) => {
        console.error('❌ WebSocket连接失败：', error);
    });
}

// 剩余函数（createOrthographicCamera/initControls/onMouseWheel/onWindowResize/toggleCameraMode）保留不变
function createOrthographicCamera(width, height) {
    const aspect = width / height;
    const baseSize = ORTHO_BASE_SIZE / (orthographicCamera?.zoom || 1);

    const orthoCam = new THREE.OrthographicCamera(
        -baseSize * aspect,
        baseSize * aspect,
        baseSize,
        -baseSize,
        0.1,
        MAX_RENDER_DISTANCE
    );
    if (!orthographicCamera) orthoCam.zoom = 1;

    orthoCam.updateProjectionMatrix();
    return orthoCam;
}

function initControls() {
    if (controls) controls.dispose();

    controls = new OrbitControls(camera, renderer.domElement);
    const config = isOrthographicMode ? CONTROLS_CONFIG.orthographic : CONTROLS_CONFIG.perspective;

    controls.minDistance = config.minDistance;
    controls.maxDistance = config.maxDistance;
    controls.enableZoom = config.enableZoom;
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.target.set(0, 1, 0);
    controls.update();
}

function onMouseWheel(event) {
    if (!isOrthographicMode) return;

    event.preventDefault();
    const delta = event.deltaY || event.deltaX;
    if (!delta) return;

    let newZoom = orthographicCamera.zoom;
    newZoom += delta > 0 ? -ORTHO_ZOOM_CONFIG.speed : ORTHO_ZOOM_CONFIG.speed;
    newZoom = Math.max(ORTHO_ZOOM_CONFIG.minZoom, Math.min(newZoom, ORTHO_ZOOM_CONFIG.maxZoom));

    orthographicCamera.zoom = newZoom;
    const container = document.getElementById('vis3d');
    const width = container.offsetWidth || 1200;
    const height = width * 9 / 16;
    const aspect = width / height;
    const baseSize = ORTHO_BASE_SIZE / newZoom;
    orthographicCamera.left = -baseSize * aspect;
    orthographicCamera.right = baseSize * aspect;
    orthographicCamera.top = baseSize;
    orthographicCamera.bottom = -baseSize;

    orthographicCamera.updateProjectionMatrix();
    console.log('正交相机参数：', {
        zoom: orthographicCamera.zoom,
        bottom: orthographicCamera.bottom,
        top: orthographicCamera.top
    });
}

// 修复：onWindowResize（移除参数）
function onWindowResize() {
    const container = document.getElementById('vis3d');
    const width = container.offsetWidth || 1200;
    const height = width * 9 / 16;

    renderer.setSize(width, height);

    perspectiveCamera.aspect = width / height;
    perspectiveCamera.updateProjectionMatrix();

    if (isOrthographicMode) {
        orthographicCamera = createOrthographicCamera(width, height);
        camera = orthographicCamera;
    }
}

window.toggleCameraMode = function () {
    const container = document.getElementById('vis3d');
    const cameraSwitchBtn = document.getElementById('cameraSwitchBtn');
    if (!container || !cameraSwitchBtn) return;

    isOrthographicMode = !isOrthographicMode;
    const width = container.offsetWidth;
    const height = container.offsetHeight || width * 9 / 16;

    const position = camera.position.clone();
    const target = controls.target.clone();

    if (isOrthographicMode) {
        orthographicCamera = createOrthographicCamera(width, height);
        camera = orthographicCamera;
        cameraSwitchBtn.innerHTML = '<i class="fas fa-cubes"></i>';
        cameraSwitchBtn.title = '切换到透视视角';
    } else {
        camera = perspectiveCamera;
        perspectiveCamera.aspect = width / height;
        perspectiveCamera.updateProjectionMatrix();
        cameraSwitchBtn.innerHTML = '<i class="fas fa-cube"></i>';
        cameraSwitchBtn.title = '切换到正交视角';
    }

    camera.position.copy(position);
    camera.lookAt(target);

    initControls();
    controls.target.copy(target);
    controls.update();

    console.log('切换到', isOrthographicMode ? '正交' : '透视', '模式');
};

// 动捕状态切换（保留不变）
window.switchMocapState = async function switchMocapState() {
    const mocapBtn = document.getElementById('mocapSwitchBtn');
    if (!mocapBtn) {
        alert('未找到动捕控制按钮！');
        return;
    }

    mocapBtn.disabled = true;

    try {
        const res = await fetch(isMocapActive ? "/mocap/stop" : "/mocap/start");
        const data = await res.json();

        if (data.code === 200) {
            isMocapActive = !isMocapActive;

            if (isMocapActive) {
                mocapBtn.classList.add('active');
                mocapBtn.innerHTML = '<i class="fas fa-stop"></i><span class="btn-text">停止动捕</span>';
                // 重新连接WebSocket（可选）
                if (socket) socket.emit('request_latest_data');
            } else {
                mocapBtn.classList.remove('active');
                mocapBtn.innerHTML = '<i class="fas fa-play"></i><span class="btn-text">启动动捕</span>';
                updateSkeleton(SMPL_TPOSE_JOINTS);
            }

            const tip = document.createElement('div');
            tip.style = "position: absolute; top: 70px; left: 20px; z-index: 100; padding: 8px 12px; background: rgba(0,0,0,0.8); color: white; border-radius: 4px; font-size: 14px;";
            tip.innerText = data.msg;
            document.getElementById('vis3d').appendChild(tip);
            setTimeout(() => tip.remove(), 2000);

            console.log("动捕状态切换结果：", data);
        } else {
            alert("操作失败：" + data.msg);
        }
    } catch (error) {
        console.error("动捕状态切换出错：", error);
        alert("操作失败：网络异常或服务端未响应");
    } finally {
        mocapBtn.disabled = false;
    }
};

// 初始化按钮状态（保留）
async function initMocapState() {
    const mocapBtn = document.getElementById('mocapSwitchBtn');
    if (!mocapBtn) return;

    try {
        const res = await fetch("/mocap/status");
        const data = await res.json();
        if (data.code === 200) {
            isMocapActive = data.data.is_listening;
            if (isMocapActive) {
                mocapBtn.classList.add('active');
                mocapBtn.innerHTML = '<i class="fas fa-stop"></i><span class="btn-text">停止动捕</span>';
            }
        }
    } catch (error) {
        console.error("获取动捕初始状态失败：", error);
    }
}

async function getMocapStatus() {
    const res = await fetch("/mocap/status");
    const data = await res.json();
    console.log("当前状态：", data);
    return data.data;
}

// 页面加载（保留）
document.addEventListener('DOMContentLoaded', function () {
    init();
    initMocapState();
});