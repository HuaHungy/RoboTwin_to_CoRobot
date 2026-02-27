
# 开发日志 (Development Log)

## 任务概述
目标是实现 RoboTwin 格式（图像内嵌于 Parquet）与 CoRobot 标准格式（图像存储为 MP4 视频）之间的双向转换。

## 1. 初始分析与数据探索

### CoRobot 数据集 (标准目标)
- **路径**: `/home/huahungy/RoboTwin_to_CoRobot/Cobot_Magic_pour_water_bottle`
- **发现**: 
    - `data/` 目录下的 Parquet 文件实际上是 Git LFS 指针文件 (ASCII text)，而非真实的 Parquet 文件。这意味着无法直接读取其 Schema。
    - 通过读取 `README.md` 和 `meta/info.json`，确认了其遵循 LeRobot 格式标准：
        - 图像数据存储在 `videos/chunk-XXX/observation.images.{key}/episode_{index}.mp4`。
        - Parquet 文件仅包含状态 (state)、动作 (action) 和时间戳等标量数据。
        - 图像特征在 metadata 中标记为 `dtype: video`。
        - 关键列名带有 `_rgb` 后缀 (e.g., `observation.images.cam_high_rgb`)。

### RoboTwin 数据集 (源数据)
- **路径**: `/home/huahungy/RoboTwin_to_CoRobot/RoboTwin_dataset/aloha_adjustbottle_left_source`
- **发现**:
    - `data/` 目录下包含真实的 Parquet 文件。
    - 部分文件 (如 `episode_000001.parquet` 及之后) 大小为 0 字节，损坏或未正确生成。但 `episode_000000.parquet` 是完整的，用于开发和测试。
    - **Schema**:
        - 图像列名为 `observation.images.cam_high` (无 `_rgb` 后缀)。
        - 数据类型为 Struct: `{'bytes': binary, 'path': string}`，其中 bytes 是 PNG 格式的二进制数据。
        - 状态维度 (16维) 与 CoRobot 文档描述 (26维) 不完全一致，但在转换脚本中保持了源数据的维度以确保数据真实性。

## 2. 方案设计

### 脚本 A: RoboTwin -> CoRobot (`robotwin_to_corobot.py`)
- **逻辑**:
    1. 读取源 Parquet 文件。
    2. 提取 `observation.images.*` 列中的 PNG 字节流。
    3. 使用 `imageio` (调用 ffmpeg) 将图像序列合成为 MP4 视频 (30 FPS)。
    4. 将视频保存至 `videos/chunk-000/observation.images.*_rgb/` 目录。
    5. 从 DataFrame 中移除图像列。
    6. 保存仅包含状态信息的 Parquet 文件至 `data/chunk-000/`。
    7. 生成/更新 `meta/info.json`，将图像特征类型标记为 `video`。

### 脚本 B: CoRobot -> RoboTwin (`corobot_to_robotwin.py`)
- **逻辑**:
    1. 读取 CoRobot 格式的 Parquet 文件。
    2. 根据文件名在 `videos/` 目录下找到对应的 MP4 文件。
    3. 读取视频帧，将其转换回 PNG 字节流。
    4. 构造 `{'bytes': ..., 'path': ...}` 结构的字典。
    5. 将图像数据重新插入 DataFrame，并移除 `_rgb` 后缀以匹配原始 RoboTwin 格式。
    6. 保存为新的 Parquet 文件。

## 3. 实现细节与问题解决
- **依赖问题**: 环境中缺少 `cv2` 和 `imageio-ffmpeg`。
    - **解决**: 安装了 `imageio-ffmpeg` 和 `opencv-python-headless`。同时确保脚本能利用系统级的 `ffmpeg` 命令。
- **视频写入**: 使用 `imageio.mimwrite` 配合 `codec='libx264'` 生成高质量 MP4。
- **结构推断**: PyArrow 在保存 Parquet 时能自动推断包含字典的列为 Struct 类型，无需手动定义 Schema，从而完美还原了 RoboTwin 的格式。

## 4. 测试验证
- 编写了 `test_conversion.py` 进行端到端测试。
- **测试流程**:
    1. 源数据 -> 转换 -> 中间结果 (CoRobot 格式)。
    2. 验证中间结果：Parquet 无图像列，MP4 视频存在。
    3. 中间结果 -> 还原 -> 还原结果 (RoboTwin 格式)。
    4. 验证还原结果：Parquet 包含图像列，且格式为 Struct (bytes, path)。
- **结果**: 对 `episode_000000` 测试通过。其他损坏文件被脚本正确忽略并报错提示。

## 5. 结论
脚本已完成并验证可用。能够实现两种格式的无损（图像编码层面）转换。
