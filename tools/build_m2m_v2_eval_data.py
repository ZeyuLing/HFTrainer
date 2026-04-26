#!/usr/bin/env python3
"""Build M2M v2 evaluation datalists from Private motion capture data.

Selects diverse actions from /apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private/
for each evaluation task (E2-E13, excluding E1 T2M which already has its own data).

Each NPZ contains: poses (T, 156), trans (T, 3), betas (1, 16), mocap_framerate.

Changes from v1:
- E9 now uses REAL low-quality motions from m2m_database quality checker (not synthetic corruption)
- All datalists include English captions (caption_en) alongside Chinese action names
- Finer-grained per-task action filtering with specific examples documented
- Each task has carefully curated action type preferences

Usage:
    python3 tools/build_m2m_v2_eval_data.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SRC_BASE = '/apdcephfs_cq11/share_1467498/home/chingshuai/HYMotion/data/npz_split/Private'
LOW_QUALITY_JSON = 'data/hymotion_m2m_refine_data/data_quality_list/low_quality.json'
OUT_DIR = 'data/eval/m2m_v2'

# -----------------------------------------------------------------------
# Chinese -> English action translation dictionary
# Covers all common action names from Private mocap data
# -----------------------------------------------------------------------
ACTION_EN_MAP = {
    # Sports - Ball
    '投篮': 'shoot a basketball', '上篮': 'layup', '灌篮': 'slam dunk',
    '运球': 'dribble a basketball', '传球': 'pass a ball', '抢篮板': 'grab a rebound',
    '射门': 'shoot (soccer)', '头球': 'header (soccer)', '颠球': 'juggle a ball',
    '发球': 'serve a ball', '扣球': 'spike a ball', '垫球': 'bump a ball',
    '足球': 'soccer', '篮球': 'basketball', '排球': 'volleyball',
    '棒球': 'baseball', '网球': 'tennis', '乒乓球': 'table tennis',
    '羽毛球': 'badminton',
    # Sports - Other
    '高尔夫': 'golf swing', '保龄球': 'bowling', '滑雪': 'skiing',
    '冲浪': 'surfing', '游泳': 'swimming', '跳远': 'long jump',
    '跳高': 'high jump', '标枪': 'javelin throw', '铁饼': 'discus throw',
    '铅球': 'shot put', '深蹲': 'squat', '硬拉': 'deadlift',
    '俯卧撑': 'push-up', '波比跳': 'burpee', '开合跳': 'jumping jack',
    '高抬腿': 'high knees', '杠铃': 'barbell exercise', '弯举': 'bicep curl',
    '平板支撑': 'plank', '弓步': 'lunge',
    # Combat
    '刺击': 'thrust attack', '砍': 'slash', '斩': 'cleave', '劈': 'chop',
    '肘击': 'elbow strike', '射击': 'shooting', '格挡': 'block/parry',
    '蓄力': 'charge attack', '冲锋': 'charge forward',
    # Daily - Standing
    '走路': 'walk', '跑': 'run', '踏步': 'march in place',
    '鞠躬': 'bow', '敬礼': 'salute', '挥手': 'wave hand',
    '握手': 'handshake', '拥抱': 'hug', '点头': 'nod',
    '摇头': 'shake head', '耸肩': 'shrug', '拍手': 'clap',
    '欢呼': 'cheer',
    # Daily - Object
    '切菜': 'chop vegetables', '炒菜': 'stir-fry',
    '扫地': 'sweep the floor', '拖地': 'mop the floor',
    '叠衣': 'fold clothes', '洗碗': 'wash dishes',
    '倒垃圾': 'take out trash',
    # Gesture
    '比心': 'heart gesture', '点赞': 'thumbs up', '胜利': 'victory sign',
    '飞吻': 'blow a kiss', '合十': 'prayer hands', '抱拳': 'fist salute',
    # Expression
    '大笑': 'laugh', '哭': 'cry', '愤怒': 'angry', '害羞': 'shy',
    '惊讶': 'surprised', '紧张': 'nervous', '疲惫': 'tired',
    # Sitting
    '坐着': 'sitting', '坐姿': 'seated pose',
    # Performance
    '唱歌': 'singing', '话剧': 'theater acting', '说唱': 'rapping',
    # Game/Hobby
    '跳房子': 'hopscotch', '跳皮筋': 'jump rope game',
    '麻将': 'mahjong', '棋': 'chess/board game',
    # Grooming
    '刷牙': 'brush teeth', '梳头': 'comb hair', '化妆': 'apply makeup',
    '剪指甲': 'clip nails', '系鞋带': 'tie shoelaces',
    # Water Sports
    '划皮划艇': 'kayak', '划桨板': 'paddleboard', '帆船': 'sailing',
    # Writing
    '写字': 'writing', '毛笔': 'calligraphy brush', '白板': 'whiteboard',
    # -----------------------------------------------------------------------
    # Full-sentence descriptive action names (from Private mocap data)
    # -----------------------------------------------------------------------
    # Combat - detailed
    '下压盾反': 'shield bash with downward press',
    '为手持炮进行折管式装填。': 'break-action reload a hand cannon',
    '半蹲双手持手枪瞄准正前方。': 'crouch and aim a pistol forward with both hands',
    '单手举飞镖在脸前瞄准不投出。': 'hold a dart up near face and aim without throwing',
    '双手持手枪做一次换弹动作。': 'reload a pistol held with both hands',
    '右肩肩扛长枪瞄准正前方。': 'shoulder a rifle on right shoulder and aim forward',
    '右膝跪地肩扛长枪瞄准前方。': 'kneel on right knee and aim a shoulder-mounted rifle',
    '手枪瞄准': 'aim a pistol',
    '把手枪收回胸前检查侧面。': 'holster pistol to chest and inspect the side',
    '把长枪收近身体做一次装填弹匣动作。': 'bring rifle close and reload the magazine',
    '扇形撒出暗器': 'fan-throw hidden weapons',
    '投掷暗器': 'throw hidden weapons',
    '从腰侧取飞镖并举起进入瞄准。': 'draw a dart from waist and aim',
    '投出飞镖后保持随势定格。': 'throw a dart and hold the follow-through',
    '投掷飞镖，举镖停顿后再投出。': 'throw a dart after pausing with it raised',
    '投掷飞镖，从胸前低位起手向前投出。': 'throw a dart forward from a low chest position',
    '投掷飞镖，向左前方出手。': 'throw a dart toward the front-left',
    '身体轻微左转后投掷飞镖。': 'turn slightly left then throw a dart',
    '侧臂投掷回旋镖': 'sidearm throw a boomerang',
    '大力投掷回旋镖': 'throw a boomerang with force',
    '投掷回旋镖': 'throw a boomerang',
    '接住回旋镖': 'catch a boomerang',
    # Daily - interaction
    '从后面拍一下别人的肩膀。': "tap someone's shoulder from behind",
    '做一个抓握式的拍肩动作。': 'grab-and-pat shoulder gesture',
    '鼓励地拍拍肩膀。': 'pat shoulder encouragingly',
    '单膝跪地行一个吻手礼。': 'kneel on one knee and kiss hand in greeting',
    '招手让别人过来。': 'beckon someone to come over',
    # Daily - object manipulation
    '单手把一个小摆件放回高架': 'place a small ornament back on a high shelf with one hand',
    '单手把水杯放在桌上': 'place a cup on the table with one hand',
    '单手拿起桌上的水杯': 'pick up a cup from the table with one hand',
    '单手费力地从高处拿下一个包裹': 'struggle to reach down a package from a high shelf',
    '双手从地上搬起一个重箱子': 'lift a heavy box from the ground with both hands',
    '双手合掌式接住': 'catch with both palms together',
    '双手把一锅汤端上桌': 'carry a pot of soup to the table with both hands',
    '双手端起桌上的一盆植物': 'pick up a potted plant from the table with both hands',
    '接住': 'catch an object',
    '放下': 'put down an object',
    '站着在书架前取书': 'reach for a book from a bookshelf while standing',
    '擦拭台面': 'wipe down a countertop',
    '按压蒸汽按钮': 'press the steam button on an iron',
    '指着一个东西并敲几下。': 'point at something and tap it a few times',
    # Daily - clothing
    '向上脱卫衣': 'pull off a hoodie overhead',
    '套头穿卫衣': 'pull on a hoodie overhead',
    '扣上衣服扣子': 'button up a shirt',
    '解开衣服扣子': 'unbutton a shirt',
    # Game / Hobby
    '从牌墙摸一张牌并审视': 'draw a tile from the wall and examine it',
    '摸一张牌后不看并打出': 'draw a tile without looking and discard it',
    '摸牌后将牌拍在桌上': 'draw a tile and slap it on the table',
    '摇骰子': 'shake dice',
    '洗卡牌': 'shuffle playing cards',
    '玩手柄': 'play with a game controller',
    '扔纸飞机': 'throw a paper airplane',
    '扔纸飞机，举起对准前方停顿后再投。': 'throw a paper airplane after pausing to aim forward',
    '扔纸飞机，从肩上高位平抛出去。': 'throw a paper airplane from a high overhand position',
    '扔纸飞机，出手时抬头看纸飞机飞向上方。': 'throw a paper airplane and look up as it flies',
    '扔纸飞机，快速甩臂把纸飞机掷出去。': 'throw a paper airplane with a quick arm whip',
    '扔纸飞机，投出后保持出手姿势定格。': 'throw a paper airplane and hold the follow-through pose',
    '扔纸飞机，用慢节奏轻轻投出去。': 'throw a paper airplane gently with slow motion',
    '扔纸飞机，身体侧对目标方向投出。': 'throw a paper airplane sideways toward the target',
    '鞭打陀螺': 'whip a spinning top',
    '站着玩VR做出投掷动作': 'play VR and make a throwing motion while standing',
    '站着玩VR拳击游戏': 'play VR boxing while standing',
    '站着玩VR音乐节奏游戏': 'play VR rhythm game while standing',
    # Gesture - detailed
    '做一个倒置的OK手势。': 'make an upside-down OK gesture',
    '做一个发射手指心的动作。': 'finger heart shooting gesture',
    '做一个展示的摊手动作。': 'present with open palms',
    '做一个把手放在耳后听的动作。': 'cup hand behind ear to listen',
    '做一个握拳庆祝的动作。': 'fist pump celebration',
    '做一个放在眼前的OK手势。': 'make an OK gesture in front of eyes',
    '做出数字3的手势并保持。': 'hold up three fingers',
    '做出数字4的手势并保持。': 'hold up four fingers',
    '手势计数，从2变到3。': 'count from two to three on fingers',
    '手势计数，从3变到4。': 'count from three to four on fingers',
    '强调第一点，做出数字1的手势。': 'hold up index finger to emphasize the first point',
    '列举全部五点，做出数字5手势后展开陈述。': 'count to five on fingers while presenting points',
    '列举第二点，做出数字2手势后展开陈述。': 'hold up two fingers while presenting second point',
    '列举第四点，做出数字4手势后展开陈述。': 'hold up four fingers while presenting fourth point',
    '低头看着摊开的手。': 'look down at open palms',
    '握个拳头。': 'clench a fist',
    '用力握紧拳头。': 'clench fist tightly',
    '用双手做出拇指朝下的手势。': 'thumbs down with both hands',
    '用大拇指指一下后面。': 'point backward with thumb',
    '用拳头敲击另一只手掌。': 'punch one fist into the other palm',
    '夸张地摊开双手。': 'spread both hands wide dramatically',
    # Expression - detailed
    '傲慢地手臂抱胸。': 'cross arms arrogantly',
    '沮丧地垂下头。': 'hang head in frustration',
    '骄傲地抬起头。': 'raise head proudly',
    '挑衅地向前倾身体。': 'lean forward provocatively',
    '挑衅地挺起胸膛。': 'puff out chest provocatively',
    '表演一个恐惧退缩的动作。': 'act out recoiling in fear',
    '表演一个无声的震惊反应。': 'act out a silent shock reaction',
    '表演一个轻蔑冷笑的姿态。': 'sneer contemptuously',
    '快速向旁边瞥一眼。': 'quickly glance to the side',
    '快速耸一下肩。': 'quick shrug',
    # Locomotion
    '像士兵一样走正步。': 'march like a soldier',
    '像孩子一样跺脚。': 'stomp feet like a child',
    '做一个左右并步移动。': 'shuffle step side to side',
    '走钢丝': 'walk a tightrope',
    '踱步': 'pace back and forth',
    '醉步': 'stagger drunkenly',
    '向后踢一下腿。': 'kick one leg backward',
    '站着抖一抖腿放松。': 'shake legs to relax while standing',
    '用脚打拍子。': 'tap foot to the beat',
    '张开手臂保持平衡。': 'spread arms to keep balance',
    '手臂展开拉伸': 'stretch with arms spread out',
    # Grooming - detailed
    '剪脚趾甲': 'clip toenails',
    '手动刮胡子': 'shave manually with a razor',
    '电动刮胡子': 'shave with an electric razor',
    '画眉毛': 'draw eyebrows with a pencil',
    '擦护肤品': 'apply skincare product',
    '用浴刷刷背': 'scrub back with a bath brush',
    '刷鞋': 'brush shoes',
    '仰头洗头发': 'wash hair with head tilted back',
    '抬头漱口': 'gargle with head tilted back',
    '弯腰刷马桶': 'bend over to scrub the toilet',
    # Writing - detailed
    '使用吸墨器给钢笔上墨': 'fill a fountain pen with an ink converter',
    '蘸墨并调整笔锋': 'dip brush in ink and adjust the tip',
    '思考时无意识地转笔': 'spin a pen absent-mindedly while thinking',
    '阅读时用手指着字': 'point at words while reading',
    # Performance / Sports detail
    '做一个演唱抒情慢歌的动作。': 'sing a slow ballad',
    '做多次正手攻球的空挥练习。': 'practice forehand swings without a ball',
    '做一个上网、后退击球后的快速中心还原步。': 'quick recovery step after net approach and backpedal shot',
    '拍照': 'take a photo',
}


def translate_action(action_name: str) -> str:
    """Translate Chinese action name to English.

    Uses keyword matching — if a known keyword appears in the action name,
    use its English translation. Otherwise return a romanized placeholder.
    For long descriptive names, try to extract the core action.
    """
    # Direct match
    if action_name in ACTION_EN_MAP:
        return ACTION_EN_MAP[action_name]
    # Keyword match (longest match first)
    best_match = ''
    best_en = ''
    for zh, en in ACTION_EN_MAP.items():
        if zh in action_name and len(zh) > len(best_match):
            best_match = zh
            best_en = en
    if best_en:
        return best_en
    # Fallback: return original (evaluation script will need manual check)
    return action_name


# -----------------------------------------------------------------------
# Action categories for diverse sampling
# -----------------------------------------------------------------------
CATEGORY_KEYWORDS = {
    'sports_ball': ['足球', '篮球', '排球', '棒球', '网球', '乒乓球', '羽毛球',
                    '投篮', '射门', '传球', '头球', '颠球', '灌篮', '抢篮板',
                    '上篮', '扣球', '发球', '垫球'],
    'sports_other': ['高尔夫', '保龄球', '滑雪', '冲浪', '划桨', '游泳',
                     '跳远', '跳高', '标枪', '铁饼', '铅球', '弓步', '波比跳',
                     '开合跳', '深蹲', '硬拉', '弯举', '平板支撑', '俯卧撑',
                     '高抬腿', '高翻', '杠铃'],
    'combat': ['刺击', '砍', '斩', '劈', '肘击', '盾', '弓', '暗器', '飞镖',
               '手枪', '长枪', '手持炮', '射击', '弹匣', '格挡', '蓄力',
               '手雷', '冲锋'],
    'daily_stand': ['走路', '跑', '踏步', '踱步', '漫步', '阔步', '醉步',
                    '蹒跚', '鞠躬', '敬礼', '挥手', '招手', '握手', '拥抱',
                    '点头', '摇头', '耸肩', '歪头', '拍手', '欢呼'],
    'daily_object': ['拿起', '放下', '搬', '提', '拎', '端', '扔', '拉',
                     '推', '切菜', '炒菜', '洗碗', '扫地', '拖地', '擦',
                     '刷', '叠衣', '晾衣', '熨烫', '倒垃圾'],
    'gesture': ['手势', 'OK', '点赞', '比心', '胜利', '飞吻', '竖起',
                '摊手', '合十', '抱拳', '指', '数字'],
    'expression': ['大笑', '哭', '愤怒', '害羞', '惊讶', '紧张', '疲惫',
                   '沮丧', '骄傲', '傲慢', '挑衅', '恐惧', '尴尬'],
    'locomotion': ['走路', '跑', '跑步', '跑动', '散步', '慢跑', '快走',
                   '冲刺', '奔跑', '正步', '跨步', '走钢丝', '跳房子',
                   '蛇形跑', '折返跑', '侧身跑'],
    'sitting': ['坐着', '坐姿'],
    'kneeling_squat': ['跪', '蹲', '下蹲'],
    'performance': ['唱歌', '演唱', '话剧', '表演', '说唱', '舞', '跳'],
    'game_hobby': ['跳房子', '跳皮筋', '陀螺', '纸飞机', '飞盘', '回旋镖',
                   '麻将', '棋', '骰子', '牌', 'VR', '手柄', '游戏'],
    'grooming': ['刷牙', '梳头', '刮胡', '化妆', '护肤', '洗头', '漱口',
                 '剪指甲', '剪脚趾', '穿', '脱', '拉链', '扣子', '系鞋带'],
    'water_sport': ['划皮划艇', '划桨板', '帆船', '冲浪'],
    'writing': ['写字', '毛笔', '钢笔', '白板', '签名', '研墨', '蘸墨'],
}


def categorize_action(action_name: str) -> str:
    """Assign an action to a category based on keywords.

    NOTE: 'expression' category is suppressed — SMPL-22 has no facial
    articulation, so expression-only clips (laugh, cry, surprise) appear
    as static standing. These are reclassified as 'other'.
    'locomotion' is checked BEFORE 'daily_stand' to avoid walk/run being
    absorbed into the generic daily category.
    """
    # Check locomotion first (higher priority than daily_stand which also has 走/跑)
    for kw in CATEGORY_KEYWORDS.get('locomotion', []):
        if kw in action_name:
            return 'locomotion'
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if cat in ('expression', 'locomotion'):
            continue  # expression suppressed, locomotion already handled
        for kw in keywords:
            if kw in action_name:
                return cat
    return 'other'


# -----------------------------------------------------------------------
# Finger-only gesture filter (SMPL-22 has no finger articulation)
# These gestures are defined entirely by finger shapes (OK, V, counting,
# thumbs up/down, etc.) and appear as static standing in SMPL-22.
# We keep gestures that involve clear arm/body movement.
# -----------------------------------------------------------------------
FINGER_ONLY_KEYWORDS = [
    'OK手势', '胜利手势', '数字1', '数字2', '数字3', '数字4', '数字5',
    '手势计数', '指尖塔', '拇指朝下', '点赞', '竖起',
    '发射手指心', '比心',  # finger heart = finger-only
    'V字手势', '数字手势', '从1数到', '从2数到', '从3数到',  # V-sign, counting
    '食指转球',  # spinning ball on fingertip
]


def is_finger_only_gesture(action_name: str, category: str) -> bool:
    """Check if a gesture action relies solely on finger articulation.

    Returns True for gestures that are invisible in SMPL-22 (no fingers).
    Expression actions are always OK (they use body/head/torso).
    """
    if category != 'gesture':
        return False
    for kw in FINGER_ONLY_KEYWORDS:
        if kw in action_name:
            return True
    return False


def scan_all_motions() -> List[Dict]:
    """Scan all NPZ files and return metadata."""
    all_motions = []
    for d in sorted(os.listdir(SRC_BASE)):
        dp = os.path.join(SRC_BASE, d)
        if not os.path.isdir(dp):
            continue
        for f in sorted(os.listdir(dp)):
            if not f.endswith('.npz'):
                continue
            path = os.path.join(dp, f)
            # Extract action name
            name = f.replace('.npz', '')
            name_clean = re.sub(r'_originalframes_\d+_\d+$', '', name)
            name_clean = re.sub(r'_take_\d+$', '', name_clean)

            try:
                data = np.load(path, allow_pickle=True)
                T = data['poses'].shape[0]
                fps = float(data.get('mocap_framerate', 30))
            except Exception:
                continue

            all_motions.append({
                'path': path,
                'rel_dir': d,
                'filename': f,
                'action_name': name_clean,
                'caption_en': translate_action(name_clean),
                'category': categorize_action(name_clean),
                'num_frames': T,
                'fps': fps,
                'duration_sec': round(T / fps, 2),
            })

    # Filter out finger-only gestures (invisible in SMPL-22)
    n_before = len(all_motions)
    all_motions = [m for m in all_motions
                   if not is_finger_only_gesture(m['action_name'], m['category'])]
    n_filtered = n_before - len(all_motions)
    if n_filtered > 0:
        print(f'  Filtered {n_filtered} finger-only gestures (invisible in SMPL-22)')

    return all_motions


def select_diverse(
    motions: List[Dict],
    n: int,
    min_frames: int = 60,
    max_frames: int = 600,
    max_per_action: int = 2,
) -> List[Dict]:
    """Select diverse motions across categories and actions."""
    # Filter by frame count
    valid = [m for m in motions if min_frames <= m['num_frames'] <= max_frames]

    # Group by category -> action
    by_cat = defaultdict(lambda: defaultdict(list))
    for m in valid:
        by_cat[m['category']][m['action_name']].append(m)

    selected = []
    rng = np.random.RandomState(42)

    # Round-robin across categories
    cat_names = sorted(by_cat.keys())
    action_queues = {}
    for cat in cat_names:
        actions = list(by_cat[cat].keys())
        rng.shuffle(actions)
        action_queues[cat] = actions

    cat_idx = 0
    while len(selected) < n:
        cat = cat_names[cat_idx % len(cat_names)]
        cat_idx += 1

        if not action_queues.get(cat):
            # Refill
            actions = list(by_cat[cat].keys())
            rng.shuffle(actions)
            action_queues[cat] = actions

        if not action_queues[cat]:
            continue

        action = action_queues[cat].pop(0)
        clips = by_cat[cat][action]
        if not clips:
            continue

        pick = clips[rng.randint(len(clips))]
        # Avoid duplicates
        if pick['path'] not in {s['path'] for s in selected}:
            selected.append(pick)

        if cat_idx > n * 10:
            break

    return selected[:n]


def build_datalist(
    motions: List[Dict],
    task_id: str,
    task_name: str,
    description: str,
    n_samples: int = 100,
    min_frames: int = 60,
    max_frames: int = 600,
    extra_fields: Optional[Dict] = None,
) -> Dict:
    """Build a datalist JSON structure."""
    selected = select_diverse(motions, n_samples, min_frames, max_frames)

    data_list = []
    for m in selected:
        item = {
            'motion_path': m['path'],
            'action_name': m['action_name'],
            'caption_en': m.get('caption_en', translate_action(m['action_name'])),
            'category': m['category'],
            'num_frames': m['num_frames'],
            'fps': m['fps'],
            'duration_sec': m['duration_sec'],
            'source': m.get('rel_dir', ''),
        }
        if extra_fields:
            item.update(extra_fields)
        data_list.append(item)

    # Category distribution
    cat_dist = defaultdict(int)
    for item in data_list:
        cat_dist[item['category']] += 1

    return {
        'meta': {
            'task_id': task_id,
            'task_name': task_name,
            'description': description,
            'total_items': len(data_list),
            'source': SRC_BASE,
            'category_distribution': dict(cat_dist),
            'min_frames': min_frames,
            'max_frames': max_frames,
        },
        'data_list': data_list,
    }


# -----------------------------------------------------------------------
# E9 specific: Load real low-quality motions from quality checker
# -----------------------------------------------------------------------
def load_low_quality_motions() -> Dict[str, List[Dict]]:
    """Load low-quality motions from m2m_database quality checker output.

    Returns dict mapping defect_type -> list of motion items.
    """
    if not os.path.exists(LOW_QUALITY_JSON):
        print(f'  ⚠️  Low-quality data not found: {LOW_QUALITY_JSON}')
        return {}

    with open(LOW_QUALITY_JSON) as f:
        data = json.load(f)

    data_dir = data.get('data_dir', 'data/hymotion_data')
    items = data['items']

    # Group by primary defect type (first in failed_checks)
    by_defect = defaultdict(list)
    for item in items:
        if not item.get('failed_checks'):
            continue
        primary_defect = item['failed_checks'][0]
        full_path = os.path.join(data_dir, item['path'])
        if os.path.exists(full_path):
            by_defect[primary_defect].append({
                'path': full_path,
                'rel_path': item['path'],
                'defect_type': primary_defect,
                'all_defects': item['failed_checks'],
            })

    return dict(by_defect)


def build_e9_repair_datalist(all_motions: List[Dict]) -> Dict:
    """Build E9: Motion Repair datalist from REAL low-quality motions.

    Strategy: Balanced sampling across ALL defect types. Each defect type
    gets up to N_per_type samples (default 15). This ensures that rare
    defect types like spine_x or knee_x are not drowned out by common
    types like foot_sliding.

    Previously: only 4 defect types × 30 samples = 120 total, with
    foot_sliding dominating (38% of samples).
    Now: all defect types × 15 samples each = ~240 total, balanced.
    """
    by_defect = load_low_quality_motions()
    if not by_defect:
        print('  ⚠️  Falling back to synthetic corruption (low-quality data unavailable)')
        return build_datalist(
            all_motions, 'E9', 'Motion Repair',
            'Repair defective motion. (FALLBACK: synthetic corruption)',
            n_samples=120, min_frames=60, max_frames=360,
        )

    N_PER_TYPE = 15  # target samples per defect type

    rng = np.random.RandomState(42)
    data_list = []
    defect_dist = {}

    # Sort defect types for reproducibility
    all_defect_types = sorted(by_defect.keys())
    print(f'    Found {len(all_defect_types)} defect types: {all_defect_types}')

    for defect_type in all_defect_types:
        candidates = by_defect[defect_type]
        if not candidates:
            continue

        # Shuffle and select up to N_PER_TYPE
        rng.shuffle(candidates)
        selected = []
        for cand in candidates:
            if len(selected) >= N_PER_TYPE:
                break
            # Load NPZ to get frame count
            try:
                npz = np.load(cand['path'], allow_pickle=True)
                T = npz['poses'].shape[0]
                fps = float(npz.get('mocap_framerate', 30))
            except Exception:
                continue

            # Filter reasonable frame count (60-600)
            if T < 60 or T > 600:
                continue

            # Extract action name from path
            fname = os.path.basename(cand['path']).replace('.npz', '')
            name_clean = re.sub(r'_originalframes_\d+_\d+$', '', fname)
            name_clean = re.sub(r'_take_\d+$', '', name_clean)

            selected.append({
                'motion_path': cand['path'],
                'action_name': name_clean,
                'caption_en': translate_action(name_clean),
                'category': f'defect_{defect_type}',
                'defect_type': defect_type,
                'all_defects': cand['all_defects'],
                'num_frames': T,
                'fps': fps,
                'duration_sec': round(T / fps, 2),
                'source': 'low_quality_db',
            })

        data_list.extend(selected)
        defect_dist[defect_type] = len(selected)
        print(f'    {defect_type}: {len(selected)}/{N_PER_TYPE} samples'
              f'{" (exhausted)" if len(selected) < N_PER_TYPE else ""}')

    # Shuffle final list for training/eval variety
    rng.shuffle(data_list)

    total = len(data_list)
    print(f'    Total: {total} samples across {len(defect_dist)} defect types')

    return {
        'meta': {
            'task_id': 'E9',
            'task_name': 'Motion Repair',
            'description': (
                'Repair REAL defective motions from quality checker. '
                'Balanced sampling: each defect type gets up to '
                f'{N_PER_TYPE} samples for fair evaluation across all '
                f'{len(defect_dist)} defect types. '
                'Model receives the defective motion as input and generates repaired version. '
                'Quality checker re-validates the output.'
            ),
            'total_items': total,
            'source': LOW_QUALITY_JSON,
            'defect_distribution': defect_dist,
            'n_per_type': N_PER_TYPE,
            'min_frames': 60,
            'max_frames': 600,
        },
        'data_list': data_list,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f'Scanning {SRC_BASE}...')
    all_motions = scan_all_motions()
    print(f'Found {len(all_motions)} motions')

    # Category stats
    cat_counts = defaultdict(int)
    for m in all_motions:
        cat_counts[m['category']] += 1
    print('Category distribution:')
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
        print(f'  {cat:20s}: {count}')

    # -----------------------------------------------------------------
    # E2: Motion In-Betweening (60-360 frames)
    #
    # Best suited actions: full-body movements with clear start/end poses
    # - sports_ball: 投篮, 上篮, 发球, 扣球 (clear start→end)
    # - daily_stand: 鞠躬, 拥抱, 握手 (social movements)
    # - combat: 刺击, 劈砍 (weapon movements)
    # - expression: 大笑, 惊讶 (emotional transitions)
    # Avoid: sitting-only, tiny gestures (too little motion to interpolate)
    # -----------------------------------------------------------------
    e2_motions = [m for m in all_motions
                  if m['category'] not in ('writing',)  # too subtle
                  and m['num_frames'] >= 60]
    e2 = build_datalist(
        e2_motions, 'E2', 'Motion In-Betweening',
        'Keep first/last N frames, generate middle. Full-body actions preferred. '
        'Settings: A=5+5 frames, B=5+5 (>200 frames only), C=30+5 (asymmetric).',
        n_samples=120, min_frames=60, max_frames=360,
    )
    save(e2, 'eval_e2_inbetween.json')

    # -----------------------------------------------------------------
    # E3: Keyframe Interpolation (90+ frames for meaningful interpolation)
    #
    # Best suited: longer sequences with continuous whole-body movement
    # - sports_other: 杠铃深蹲, 高尔夫挥杆 (repetitive full-body)
    # - combat: 连续射击, 冲锋 (sustained sequences)
    # - daily_stand: 走路, 跑步 (locomotion)
    # - performance: 舞蹈, 话剧 (expressive long sequences)
    # Needs: >90 frames for Setting A (30-frame interval), >180 for Setting B
    # -----------------------------------------------------------------
    e3 = build_datalist(
        all_motions, 'E3', 'Keyframe Interpolation',
        'Keep every K-th keyframe, interpolate rest. Longer sequences preferred. '
        'Settings: A=30f interval, B=60f, C=15f, D=random 10-90f.',
        n_samples=120, min_frames=90, max_frames=600,
    )
    save(e3, 'eval_e3_keyframe.json')

    # -----------------------------------------------------------------
    # E4: End-Effector Constraint
    #
    # Best suited: actions with clear hand/foot target positions
    # - sports_ball: 投篮(hand position matters), 射门(foot position)
    # - daily_object: 切菜, 扫地, 拖地 (hands on tools)
    # - combat: 刺击, 射击 (hand aiming)
    # - gesture: 比心, 点赞 (precise hand placement)
    # Settings test wrist-only, ankle-only, mixed, and with text+keypose
    # -----------------------------------------------------------------
    e4_prefer = ['sports_ball', 'daily_object', 'combat', 'gesture',
                 'grooming', 'writing', 'game_hobby']
    e4_motions = [m for m in all_motions if m['category'] in e4_prefer]
    # Fallback: use all if not enough
    if len([m for m in e4_motions if 60 <= m['num_frames'] <= 360]) < 80:
        e4_motions = all_motions

    e4 = build_datalist(
        e4_motions, 'E4', 'End-Effector Constraint',
        'Constrain end-effector world positions at sparse frames. '
        'Prefer actions with clear hand/foot targets. '
        'Settings: A=right wrist, B=both ankles, C=right hand+left foot, '
        'D=text+first keypose, E=text+first&last keypose.',
        n_samples=100, min_frames=60, max_frames=360,
    )
    save(e4, 'eval_e4_end_effector.json')

    # -----------------------------------------------------------------
    # E5: Trajectory Following
    #
    # MUST have significant root (pelvis) movement in XZ plane
    # Best suited: locomotion-heavy actions
    # - daily_stand: 走路, 跑步, 踏步 (pure locomotion)
    # - sports_ball: 带球跑, 助跑投篮 (locomotion + action)
    # - sports_other: 滑雪, 冲浪 (sliding locomotion)
    # - combat: 冲锋, 跑动射击 (tactical movement)
    # Exclude: sitting, stationary gestures, writing
    # -----------------------------------------------------------------
    locomotion_keywords = ['走', '跑', '踏步', '移动', '冲', '跳', '助跑', '步法',
                           '跑动', '滑步', '前进', '后退', '绕', '转身', '侧步',
                           '闪避', '翻滚', '冲刺', '滑行', '追']
    traj_motions = []
    for m in all_motions:
        if m['num_frames'] < 60 or m['num_frames'] > 600:
            continue
        # Categories with inherent root movement
        if m['category'] in ('sports_ball', 'sports_other', 'daily_stand', 'combat'):
            traj_motions.append(m)
            continue
        # Keyword-based locomotion detection
        for kw in locomotion_keywords:
            if kw in m['action_name']:
                traj_motions.append(m)
                break
    traj_motions = list({m['path']: m for m in traj_motions}.values())

    e5 = build_datalist(
        traj_motions if len(traj_motions) >= 80 else all_motions,
        'E5', 'Trajectory Following',
        'Follow root XZ trajectory. Locomotion-heavy actions only. '
        'Settings: A=dense root XZ every frame, B=sparse waypoints every 30f, '
        'C=trajectory+heading (root rotation), D=heading only every 30f.',
        n_samples=100, min_frames=60, max_frames=600,
    )
    save(e5, 'eval_e5_trajectory.json')

    # -----------------------------------------------------------------
    # E6: Foot Ground Constraint
    #
    # Tests the model's ability to respect ankle-ground contact constraints.
    # The constraint tells the model: "at these frames, the ankle should
    # be at the ground plane" — either via rotation constraint (keeping
    # the GT ankle rotation) or position constraint (ankle Y=0, or XZ lock).
    #
    # Best suited: standing/walking/running actions where feet clearly
    # alternate between ground contact and flight phase
    # - daily_stand: 走路, 跑步, 踏步 (clear gait cycles)
    # - sports_ball: 助跑投篮, 带球跑 (running + action)
    # - sports_other: 跳远, 弓步 (clear contact phases)
    # - combat: 步法移动, 格斗站位 (footwork)
    # Exclude: sitting, water_sport (no ground contact), writing
    # -----------------------------------------------------------------
    foot_categories = ('daily_stand', 'sports_ball', 'sports_other',
                       'combat', 'performance', 'game_hobby')
    foot_motions = [m for m in all_motions if m['category'] in foot_categories]

    e6 = build_datalist(
        foot_motions if len(foot_motions) >= 80 else all_motions,
        'E6', 'Foot Ground Constraint',
        'Constrain ankle at ground-contact frames. Standing/walking actions preferred. '
        'Contact frames detected from GT motion: ankle Y < 5cm = contact. '
        'Settings: A_rot=GT contact frames rotation, B_rot=all frames rotation, '
        'C_pos_y=contact Y-only, D_pos_xz=contact XZ, E_pos_xyz=contact full 3D.',
        n_samples=100, min_frames=60, max_frames=360,
    )
    save(e6, 'eval_e6_foot_ground.json')

    # -----------------------------------------------------------------
    # E7: First-Frame Continuation
    #
    # Tests continuation from a single starting pose
    # Best suited: diverse starting poses across many action types
    # - expression: 大笑, 哭泣 (emotional continuation)
    # - gesture: 比心, 点赞 (gesture from start)
    # - sports_ball: 投篮准备 (sports start)
    # - combat: 战斗起手 (combat initiation)
    # Want maximum pose diversity in frame 0
    # -----------------------------------------------------------------
    e7 = build_datalist(
        all_motions, 'E7', 'First-Frame Continuation',
        'Keep frame 0 (full-body 135d), generate rest. Caption provides action_name. '
        'Diverse starting poses across all categories.',
        n_samples=100, min_frames=60, max_frames=300,
    )
    save(e7, 'eval_e7_first_frame.json')

    # -----------------------------------------------------------------
    # E8: Loop Animation
    #
    # Tests ability to generate seamless looping animation
    # MUST use naturally cyclic/repetitive actions:
    # - sports_other: 原地踏步, 深蹲, 开合跳, 高抬腿, 波比跳
    # - daily_stand: 踏步, 走路 (gait cycle)
    # - sports_ball: 运球, 颠球 (ball bouncing)
    # - water_sport: 划桨 (paddling cycle)
    # - sitting: 抖腿, 摇椅 (repetitive sitting)
    # Exclude: one-shot actions (投篮, 跳远)
    # -----------------------------------------------------------------
    loop_keywords = ['原地', '踏步', '跑', '走', '运球', '颠球', '跳',
                     '划桨', '摇', '抖腿', '拍', '呼吸', '深蹲', '开合跳',
                     '高抬腿', '波比跳', '弹', '摆', '晃', '跺', '挥',
                     '左右', '反复']
    loop_motions = [m for m in all_motions
                    if any(kw in m['action_name'] for kw in loop_keywords)
                    and 60 <= m['num_frames'] <= 300]
    e8 = build_datalist(
        loop_motions if len(loop_motions) >= 60 else all_motions,
        'E8', 'Loop Animation',
        'Generate seamless looping animation. Naturally cyclic motions only '
        '(running, dribbling, jumping jacks, squats, paddling, etc). '
        'Settings: A=first=last frame constraint, B=loop+dense trajectory, '
        'C=loop+sparse waypoints.',
        n_samples=80, min_frames=60, max_frames=300,
    )
    save(e8, 'eval_e8_loop.json')

    # -----------------------------------------------------------------
    # E9: Motion Repair — uses REAL low-quality motions
    # -----------------------------------------------------------------
    print('\n--- E9: Motion Repair (real low-quality data) ---')
    e9 = build_e9_repair_datalist(all_motions)
    save(e9, 'eval_e9_repair.json')

    # -----------------------------------------------------------------
    # E10: Part-Level Control
    #
    # Tests regenerating one body part while keeping another fixed
    # Best suited: actions with distinct upper/lower body patterns
    # - sports_ball: 投篮 (upper: throw, lower: jump)
    # - daily_object: 扫地 (upper: sweep, lower: walk)
    # - combat: 劈砍 (upper: attack, lower: stance)
    # - performance: 舞蹈 (coordinated full-body)
    # Settings: A=keep upper regen lower, B=keep lower regen upper,
    #           C=keep root only regen all
    # -----------------------------------------------------------------
    e10 = build_datalist(
        all_motions, 'E10', 'Part-Level Control',
        'Keep one body part, regenerate rest. Prefer actions with distinct '
        'upper/lower body patterns. Settings: A=keep upper body, '
        'B=keep lower body, C=keep root (pelvis) only.',
        n_samples=100, min_frames=60, max_frames=360,
    )
    save(e10, 'eval_e10_part_control.json')

    # -----------------------------------------------------------------
    # E11: Caption Completion
    #
    # Tests text-conditioned completion (in-betweening/keyframe + caption)
    # Needs actions with clear, descriptive names that serve as captions
    # English caption is used as text condition
    # -----------------------------------------------------------------
    e11 = build_datalist(
        all_motions, 'E11', 'Caption Completion',
        'Completion with text caption. English caption (from action_name) + '
        'motion constraints. Settings: inbetween=first/last 5f + caption, '
        'keyframe=every 30f + caption.',
        n_samples=100, min_frames=60, max_frames=360,
    )
    for item in e11['data_list']:
        item['caption'] = item['caption_en']
        item['caption_zh'] = item['action_name']
        item['has_caption'] = True
    save(e11, 'eval_e11_caption_completion.json')

    # -----------------------------------------------------------------
    # E13: Multi-Prompt Generation
    #
    # Tests chaining multiple text prompts into long motion
    # Each segment uses a different action prompt
    # Needs diverse, short-medium actions that can be meaningfully chained
    # -----------------------------------------------------------------
    e13 = build_datalist(
        all_motions, 'E13', 'Multi-Prompt Generation',
        'Chain multiple text prompts into long motion via autoregressive sliding window. '
        'Each segment uses prev segment\'s last N frames as in-between condition. '
        'Settings: A=3 prompts 5f overlap, B=5 prompts 5f overlap, C=10 prompts 10f overlap.',
        n_samples=80, min_frames=60, max_frames=240,
    )
    for item in e13['data_list']:
        item['caption'] = item['caption_en']
        item['caption_zh'] = item['action_name']
        item['has_caption'] = True
    save(e13, 'eval_e13_multi_prompt.json')

    print(f'\nAll datalists saved to {OUT_DIR}/')


def save(data: Dict, filename: str):
    path = os.path.join(OUT_DIR, filename)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    n = data['meta']['total_items']
    meta = data['meta']
    if 'category_distribution' in meta:
        cats = meta['category_distribution']
        print(f'  ✅ {filename}: {n} samples, {len(cats)} categories')
    elif 'defect_distribution' in meta:
        defects = meta['defect_distribution']
        print(f'  ✅ {filename}: {n} samples, defects={defects}')


if __name__ == '__main__':
    main()
