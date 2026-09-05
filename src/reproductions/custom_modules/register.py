# -*- coding: utf-8 -*-
"""
把 [36]/[28] 的自定义模块注册进 ultralytics, 使其能被模型 YAML 解析并正确推通道。
做法(最小侵入): 取 parse_model 源码 -> 把自定义模块名注入 base_modules/repeat_modules
两个 frozenset -> 在 tasks 命名空间重新 exec。适配当前 ultralytics 8.3.x。
"""
import os, sys, inspect
import ultralytics.nn.tasks as T

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from custom_modules.blocks_36 import C2f_iRMB, ThreeChAgg
from custom_modules.blocks_28 import CSPPC, MultiDimen, SpatialConv

_ALL = [C2f_iRMB, ThreeChAgg, CSPPC, MultiDimen, SpatialConv]   # 都进 base_modules
_REPEAT = [C2f_iRMB, CSPPC]                                     # 额外进 repeat_modules
_ANCHOR = "            A2C2f,\n        }\n    )"


def register():
    for c in _ALL:                       # 1) 注入命名空间: 供 globals()[name] 与 frozenset 引用
        setattr(T, c.__name__, c)
    if getattr(T, "_custom_registered", False):
        return
    src = inspect.getsource(T.parse_model)
    if src.count(_ANCHOR) < 2:
        raise RuntimeError("注册失败: ultralytics 版本结构变化, 未命中 frozenset 锚点")
    base_names = ", ".join(c.__name__ for c in _ALL)
    rep_names = ", ".join(c.__name__ for c in _REPEAT)
    # 第一个锚点 = base_modules, 第二个 = repeat_modules
    src = src.replace(_ANCHOR, f"            A2C2f,\n            {base_names},\n        }}\n    )", 1)
    src = src.replace(_ANCHOR, f"            A2C2f,\n            {rep_names},\n        }}\n    )", 1)
    exec(compile(src, T.__file__, "exec"), T.__dict__)   # 在 tasks 命名空间重定义 parse_model
    T._custom_registered = True
    print("[register] 已注册自定义模块:", [c.__name__ for c in _ALL])


if __name__ == "__main__":
    register()
