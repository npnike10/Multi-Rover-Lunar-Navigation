from srb.utils.registry import register_srb_tasks

from .task import MarlWaypointTask, MarlWaypointTaskCfg

BASE_TASK_NAME = __name__.split(".")[-1]
register_srb_tasks(
    {
        BASE_TASK_NAME: {},
    },
    default_entry_point=MarlWaypointTask,
    default_task_cfg=MarlWaypointTaskCfg,
)
