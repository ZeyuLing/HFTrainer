/**
 * 骨架数据模块
 * 包含各种预定义的骨架结构
 */

// Three.js Vector3辅助函数
function createVector3(x, y, z) {
    return new THREE.Vector3(x, y, z);
}

// 将毫米转换为米（Three.js场景更适合米的单位）
function mmToM(value) {
    return value * 0.001;
}

/**
 * SMPL骨架数据 (24关节)
 */
const SMPL_SKELETON = {
    name: "SMPL (24关节)",
    description: "SMPL人体模型骨架，包含24个关节",
    jointCount: 24,
    jointOffsets: [
        [0.0000, 0.0000, 0.0000],          // 0: pelvis
        [0.0585813, -0.0822800, -0.0176641], // 1: left_hip
        [-0.0603097, -0.0905133, -0.0135425], // 2: right_hip
        [0.0044394, 0.1244036, -0.0383852], // 3: spine1
        [0.0434514, -0.3864695, 0.0080370], // 4: left_knee
        [-0.0432566, -0.3836879, -0.0048430], // 5: right_knee
        [0.0044884, 0.1379564, 0.0268203], // 6: spine2
        [-0.0147903, -0.4268745, -0.0374280], // 7: left_ankle
        [0.0190555, -0.4200456, -0.0345617], // 8: right_ankle
        [-0.0022646, 0.0560324, 0.0028550], // 9: spine3
        [0.0410544, -0.0602859, 0.1220424], // 10: left_foot
        [-0.0348399, -0.0621055, 0.1303233], // 11: right_foot
        [-0.0133902, 0.2116355, -0.0334676], // 12: neck
        [0.0717025, 0.1139997, -0.0188982], // 13: left_collar
        [-0.0829537, 0.1124724, -0.0237074], // 14: right_collar
        [0.0101132, 0.0889373, 0.0504099], // 15: head
        [0.1229214, 0.0452051, -0.0190460], // 16: left_shoulder
        [-0.1132283, 0.0468532, -0.0084721], // 17: right_shoulder
        [0.2553319, -0.0156490, -0.0229465], // 18: left_elbow
        [-0.2601275, -0.0143692, -0.0312687], // 19: right_elbow
        [0.2657092, 0.0126981, -0.0073747], // 20: left_wrist
        [-0.2691084, 0.0067937, -0.0060268], // 21: right_wrist
        [0.0866905, -0.0106360, -0.0155943], // 22: left_hand
        [-0.0887537, -0.0086516, -0.0101071]  // 23: right_hand
    ].map(arr => createVector3(arr[0], arr[1], arr[2])),

    kinematicTree: {
        1: [[0, 1], [0, 2], [0, 3]],
        2: [[1, 4], [2, 5], [3, 6]],
        3: [[4, 7], [5, 8], [6, 9]],
        4: [[7, 10], [8, 11], [9, 12], [9, 13], [9, 14]],
        5: [[12, 15], [13, 16], [14, 17]],
        6: [[16, 18], [17, 19]],
        7: [[18, 20], [19, 21]],
        8: [[20, 22], [21, 23]]
    },

    boneDensity: {
        "(0, 3)": 1.0,
        "(3, 6)": 0.64,
        "(6, 9)": 0.49,
        "(9, 12)": 0.16,
        "(12, 15)": 0.04,
        "(0, 1)": 0.09,
        "(1, 4)": 0.16,
        "(4, 7)": 0.04,
        "(7, 10)": 0.0,
        "(0, 2)": 0.09,
        "(2, 5)": 0.16,
        "(5, 8)": 0.04,
        "(8, 11)": 0.0,
        "(9, 13)": 0.04,
        "(13, 16)": 0.04,
        "(16, 18)": 0.0225,
        "(18, 20)": 0.01,
        "(20, 22)": 0.0,
        "(9, 14)": 0.04,
        "(14, 17)": 0.04,
        "(17, 19)": 0.0225,
        "(19, 21)": 0.01,
        "(21, 23)": 0.0
    },

    jointNames: [
        "pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
        "spine2", "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot",
        "neck", "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hand", "right_hand"
    ],

    symmetricPairs: [
        [1, 2],   // 左髋 ↔ 右髋
        [4, 5],   // 左大腿 ↔ 右大腿
        [7, 8],   // 左小腿 ↔ 右小腿
        [10, 11], // 左足 ↔ 右足
        [13, 14], // 左锁骨 ↔ 右锁骨
        [16, 17], // 左肩 ↔ 右肩
        [18, 19], // 左肘 ↔ 右肘
        [20, 21], // 左手腕 ↔ 右手腕
        [22, 23]  // 左手 ↔ 右手
    ]
};

/**
 * 简单人体骨架 (15关节)
 */
const SIMPLE_HUMAN_SKELETON = {
    name: "简单人体 (15关节)",
    description: "简化的人体骨架，包含15个主要关节",
    jointCount: 15,
    jointOffsets: [
        [0, 0, 0],        // 0: 骨盆
        [0.1, 0, 0],      // 1: 左髋
        [-0.1, 0, 0],     // 2: 右髋
        [0, 0.2, 0],      // 3: 脊柱
        [0, 0.3, 0],      // 4: 胸部
        [0, 0.4, 0],      // 5: 颈部
        [0, 0.5, 0],      // 6: 头部
        [0.1, 0.3, 0],    // 7: 左肩
        [-0.1, 0.3, 0],   // 8: 右肩
        [0.2, 0.2, 0],    // 9: 左肘
        [-0.2, 0.2, 0],   // 10: 右肘
        [0.3, 0.1, 0],    // 11: 左手腕
        [-0.3, 0.1, 0],   // 12: 右手腕
        [0, -0.3, 0],     // 13: 左膝
        [0, -0.3, 0]      // 14: 右膝
    ].map(arr => createVector3(arr[0], arr[1], arr[2])),

    kinematicTree: {
        1: [[0, 1], [0, 2], [0, 3]],
        2: [[1, 13], [2, 14], [3, 4]],
        3: [[4, 5], [4, 7], [4, 8]],
        4: [[5, 6], [7, 9], [8, 10]],
        5: [[9, 11], [10, 12]]
    },

    boneDensity: {
        "(0, 3)": 1.0,
        "(3, 4)": 0.8,
        "(4, 5)": 0.6,
        "(5, 6)": 0.4,
        "(0, 1)": 0.7,
        "(1, 13)": 0.5,
        "(0, 2)": 0.7,
        "(2, 14)": 0.5,
        "(4, 7)": 0.6,
        "(7, 9)": 0.4,
        "(9, 11)": 0.2,
        "(4, 8)": 0.6,
        "(8, 10)": 0.4,
        "(10, 12)": 0.2
    },

    jointNames: [
        "pelvis", "left_hip", "right_hip", "spine1", "chest",
        "neck", "head", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
        "left_knee", "right_knee"
    ],

    symmetricPairs: [
        [1, 2], [7, 8], [9, 10], [11, 12], [13, 14]
    ]
};

/**
 * 机械臂骨架 (6关节)
 */
const ROBOT_ARM_SKELETON = {
    name: "机械臂 (6关节)",
    description: "简单的机械臂骨架，包含6个关节",
    jointCount: 6,
    jointOffsets: [
        [0, 0, 0],        // 0: 基座
        [0, 0.3, 0],      // 1: 关节1
        [0, 0.5, 0],      // 2: 关节2
        [0.4, 0, 0],      // 3: 关节3
        [0.3, 0, 0],      // 4: 关节4
        [0.2, 0, 0]       // 5: 关节5
    ].map(arr => createVector3(arr[0], arr[1], arr[2])),

    kinematicTree: {
        1: [[0, 1]],
        2: [[1, 2]],
        3: [[2, 3]],
        4: [[3, 4]],
        5: [[4, 5]]
    },

    boneDensity: {
        "(0, 1)": 1.0,
        "(1, 2)": 0.8,
        "(2, 3)": 0.6,
        "(3, 4)": 0.5,
        "(4, 5)": 0.4
    },

    jointNames: [
        "base", "joint1", "joint2", "joint3", "joint4", "joint5"
    ],

    symmetricPairs: []
};

/**
 * 四足动物骨架 (20关节)
 */
const QUADRUPED_SKELETON = {
    name: "四足动物 (20关节)",
    description: "四足动物骨架，包含20个关节",
    jointCount: 20,
    jointOffsets: [
        [0, 0, 0],        // 0: 身体中心
        [0.2, 0.1, 0.1],  // 1: 左前腿肩
        [-0.2, 0.1, 0.1], // 2: 右前腿肩
        [0.2, -0.1, 0.1], // 3: 左后腿髋
        [-0.2, -0.1, 0.1],// 4: 右后腿髋
        [0.4, 0, 0],      // 5: 左前腿肘
        [-0.4, 0, 0],     // 6: 右前腿肘
        [0.4, -0.2, 0],   // 7: 左后腿膝
        [-0.4, -0.2, 0],  // 8: 右后腿膝
        [0.6, -0.1, 0],   // 9: 左前腿腕
        [-0.6, -0.1, 0],  // 10: 右前腿腕
        [0.6, -0.3, 0],   // 11: 左后腿踝
        [-0.6, -0.3, 0],  // 12: 右后腿踝
        [0.7, -0.2, 0],   // 13: 左前脚
        [-0.7, -0.2, 0],  // 14: 右前脚
        [0.7, -0.4, 0],   // 15: 左后脚
        [-0.7, -0.4, 0],  // 16: 右后脚
        [0, 0.2, 0],      // 17: 颈部
        [0, 0.4, 0],      // 18: 头部
        [0, 0.5, 0]       // 19: 鼻子
    ].map(arr => createVector3(arr[0], arr[1], arr[2])),

    kinematicTree: {
        1: [[0, 1], [0, 2], [0, 3], [0, 4], [0, 17]],
        2: [[1, 5], [2, 6], [3, 7], [4, 8], [17, 18]],
        3: [[5, 9], [6, 10], [7, 11], [8, 12], [18, 19]],
        4: [[9, 13], [10, 14], [11, 15], [12, 16]]
    },

    boneDensity: {
        "(0, 1)": 0.8, "(0, 2)": 0.8, "(0, 3)": 0.8, "(0, 4)": 0.8,
        "(0, 17)": 0.7, "(1, 5)": 0.6, "(2, 6)": 0.6, "(3, 7)": 0.6,
        "(4, 8)": 0.6, "(5, 9)": 0.4, "(6, 10)": 0.4, "(7, 11)": 0.4,
        "(8, 12)": 0.4, "(9, 13)": 0.2, "(10, 14)": 0.2, "(11, 15)": 0.2,
        "(12, 16)": 0.2, "(17, 18)": 0.5, "(18, 19)": 0.3
    },

    jointNames: [
        "body", "left_front_shoulder", "right_front_shoulder",
        "left_rear_hip", "right_rear_hip", "left_front_elbow",
        "right_front_elbow", "left_rear_knee", "right_rear_knee",
        "left_front_wrist", "right_front_wrist", "left_rear_ankle",
        "right_rear_ankle", "left_front_foot", "right_front_foot",
        "left_rear_foot", "right_rear_foot", "neck", "head", "nose"
    ],

    symmetricPairs: [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
        [11, 12], [13, 14], [15, 16]
    ]
};

/**
 * 骨架数据管理器
 */
class SkeletonDataManager {
    constructor() {
        this.skeletons = {
            'smpl': SMPL_SKELETON,
            'simple': SIMPLE_HUMAN_SKELETON,
            'robot': ROBOT_ARM_SKELETON,
            'quadruped': QUADRUPED_SKELETON
        };

        this.currentSkeleton = 'smpl';
    }

    /**
     * 获取所有可用的骨架
     */
    getAllSkeletons() {
        return Object.keys(this.skeletons).map(key => ({
            id: key,
            name: this.skeletons[key].name,
            description: this.skeletons[key].description,
            jointCount: this.skeletons[key].jointCount
        }));
    }

    /**
     * 获取当前骨架数据
     */
    getCurrentSkeleton() {
        return this.skeletons[this.currentSkeleton];
    }

    /**
     * 设置当前骨架
     */
    setCurrentSkeleton(skeletonId) {
        if (this.skeletons[skeletonId]) {
            this.currentSkeleton = skeletonId;
            return true;
        }
        return false;
    }

    /**
     * 获取骨架数据
     */
    getSkeleton(skeletonId) {
        return this.skeletons[skeletonId] || null;
    }

    /**
     * 创建自定义骨架
     */
    createCustomSkeleton(data) {
        return {
            name: data.name || "自定义骨架",
            description: data.description || "用户自定义骨架",
            jointCount: data.jointOffsets?.length || 0,
            jointOffsets: data.jointOffsets || [],
            kinematicTree: data.kinematicTree || {},
            boneDensity: data.boneDensity || {},
            jointNames: data.jointNames || [],
            symmetricPairs: data.symmetricPairs || []
        };
    }

    /**
     * 计算骨骼数量
     */
    calculateBoneCount(kinematicTree) {
        let count = 0;
        for (const level in kinematicTree) {
            count += kinematicTree[level].length;
        }
        return count;
    }
}

// 导出全局实例
window.SkeletonDataManager = SkeletonDataManager;
