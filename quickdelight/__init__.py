"""QuickDelight 主包。"""

# 包级别目前只公开一个简单版本号。
# 这里不主动导入其他子模块，避免用户只是 `import quickdelight`
# 时就触发较重依赖的加载。
__version__ = "0.1.0"

__all__ = ["__version__"]
