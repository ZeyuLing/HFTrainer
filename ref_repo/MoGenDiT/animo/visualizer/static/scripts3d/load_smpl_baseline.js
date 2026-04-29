import * as THREE from 'three';


const NUM_SKIN_WEIGHTS = 4;

async function load_smpl_with_shapes(shapes, gender, person_id) {
    const urls =
        {
            'neutral': [
                '/static/assets/dump_smplh/v_template.bin',
                '/static/assets/dump_smplh/faces.bin',
                '/static/assets/dump_smplh/skinWeights.bin',
                '/static/assets/dump_smplh/skinIndice.bin',
                '/static/assets/dump_smplh/keypoints.bin',
            ],
            'male': [
                '/static/assets/dump_smplh_male/v_template.bin',
                '/static/assets/dump_smplh_male/faces.bin',
                '/static/assets/dump_smplh_male/skinWeights.bin',
                '/static/assets/dump_smplh_male/skinIndice.bin',
                '/static/assets/dump_smplh_male/keypoints.bin',
            ],
            'female': [
                '/static/assets/dump_smplh_female/v_template.bin',
                '/static/assets/dump_smplh_female/faces.bin',
                '/static/assets/dump_smplh_female/skinWeights.bin',
                '/static/assets/dump_smplh_female/skinIndice.bin',
                '/static/assets/dump_smplh_female/keypoints.bin',
            ]
        }[gender];

    const baseline_colors = [
        0xffffff,  // 0 - 白色
        0x6495ED,  // 1 - 蓝色  
        0xFF6B81,  // 2 - 红色
        0x32CD32,  // 3 - 绿色
        0xFFD700,  // 4 - 金色
        0xFF69B4,  // 5 - 粉色
        0x888888,  // 6 - 灰色
        0x4CAF50,  // 7 - 深绿色
        0xFF8C00,  // 8 - 橙色
        0x9370DB,  // 9 - 紫色
    ];

    console.log(shapes.length);
    const geometry = new THREE.BufferGeometry();
    const buffers = await Promise.all(
        urls.map(url => fetch(url).then(response => response.arrayBuffer()))
    );
    const v_template = new Float32Array(buffers[0]);
    let offsets;

    // 新增：处理shapes数组的条件分支
    // 1. 先判断shapes长度是否至少为16（前10位+后6位）
    // 2. 检查最后6个元素是否全部等于0
    const isLastSixZeros = shapes.length >= 16 &&
        shapes.slice(-6).every(element => element === 0);

    if (isLastSixZeros) {
        // 满足条件：仅保留前10位元素
        shapes = shapes.slice(0, 10);
    }

    // 原有逻辑：根据处理后的shapes长度请求对应offset
    if (shapes.length === 10) {
        offsets = await Promise.all(
            shapes.map((_, i) =>
                fetch(`/static/assets/dump_smpl/shapeoffset_${i}.bin`)
                    .then(response => response.arrayBuffer())
                    .then(buffer => new Float32Array(buffer))
            )
        );
    } else {
        offsets = await Promise.all(
            shapes.map((_, i) =>
                fetch(`/static/assets/dump_smplh/shapeoffset_${i}.bin`)
                    .then(response => response.arrayBuffer())
                    .then(buffer => new Float32Array(buffer))
            )
        );
    }
    // offsets.forEach((offset, i) => {
    //     for (let j = 0; j < v_template.length / 3; j++) {
    //         v_template[3 * j] += offset[3 * j] * shapes[i];
    //         v_template[3 * j + 1] += offset[3 * j + 1] * shapes[i];
    //         v_template[3 * j + 2] += offset[3 * j + 2] * shapes[i];
    //     }
    // });
    const faces = new Uint16Array(buffers[1]);
    const skinWeights = new Float32Array(buffers[2]);
    const skinIndices = new Uint16Array(buffers[3]);
    const keypoints = new Float32Array(buffers[4]);
    // edges包含骨架链接关系
    // const edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21];
    const edges = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34, 35, 21, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50];
    // 假设 jointPositions 是一个 J x 3 的数组，每个元素是一个包含 X, Y, Z 坐标的数组
    var rootBone = new THREE.Bone();
    rootBone.position.set(keypoints[0], keypoints[1], keypoints[2]);
    // scene.add(rootBone);
    var bones = [rootBone];
    // 创建骨骼
    for (let i = 1; i < keypoints.length / 3; i++) {
        const bone = new THREE.Bone();
        const parentIndex = edges[i];
        bone.position.set(
            keypoints[3 * i] - keypoints[3 * parentIndex],
            keypoints[3 * i + 1] - keypoints[3 * parentIndex + 1],
            keypoints[3 * i + 2] - keypoints[3 * parentIndex + 2]);
        console.log(i, bone.position);
        bones.push(bone);
        bones[parentIndex].add(bone);
    }
    var skeleton = new THREE.Skeleton(bones);
    geometry.setIndex(new THREE.BufferAttribute(faces, 1));

    geometry.setAttribute('position', new THREE.BufferAttribute(v_template, 3));
    geometry.setAttribute('skinIndex', new THREE.BufferAttribute(skinIndices, NUM_SKIN_WEIGHTS));
    geometry.setAttribute('skinWeight', new THREE.BufferAttribute(skinWeights, NUM_SKIN_WEIGHTS));

    geometry.computeVertexNormals();
    console.log(geometry);
    // const material = new THREE.MeshBasicMaterial({ color: 0x00ff00, skinning: true, side: THREE.DoubleSide, });
    const material = new THREE.MeshStandardMaterial({
        color: baseline_colors[person_id % baseline_colors.length],
        skinning: true,
        side: THREE.DoubleSide
    });
    var mesh = new THREE.SkinnedMesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.add(bones[0]);
    mesh.bind(skeleton);
    bones[0].rotation.x = Math.PI / 2;
    bones[0].position.z = 1.1;
    return { bones, skeleton, mesh };
}

function reshapeArrayTo2D(float32Array, rows) {
    const twoDArray = [];
    const cols = float32Array.length / rows;
    for (let i = 0; i < rows; i++) {
        const row = new Float32Array(cols);
        for (let j = 0; j < cols; j++) {
            row[j] = float32Array[i * cols + j];
        }
        twoDArray.push(row);
    }
    return twoDArray;
}


export { load_smpl_with_shapes };
