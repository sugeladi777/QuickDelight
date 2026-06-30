from __future__ import annotations

"""QuickDelight 包级命令入口。"""

# 这个文件保持极薄。
# 作用只有一个：支持 `python -m quickdelight ...` 的启动方式。
# 真正的参数解析、子命令注册、执行分发都在 `quickdelight/cli.py` 中。
from quickdelight.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
