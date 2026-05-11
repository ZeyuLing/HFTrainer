try:
    from hftrainer.datasets.motion.motionhub.single_agent_dataset import MotionHubSingleAgentDataset
except Exception:
    MotionHubSingleAgentDataset = None
try:
    from hftrainer.datasets.motion.motionhub.single_agent_text_dataset import MotionHubSingleAgentTextDataset
except Exception:
    MotionHubSingleAgentTextDataset = None
try:
    from hftrainer.datasets.motion.motionhub.multitask_multiagent_dataset import MotionhubMultiTaskMultiAgentDataset
except Exception:
    MotionhubMultiTaskMultiAgentDataset = None

__all__ = [
    'MotionHubSingleAgentDataset',
    'MotionHubSingleAgentTextDataset',
    'MotionhubMultiTaskMultiAgentDataset',
]
