
# RoboTwin 与 CoRobot 数据集格式转换工具

本项目提供了两个 Python 脚本，用于在 **RoboTwin 格式**（图像以 PNG 二进制流内嵌于 Parquet 文件）与 **CoRobot 标准格式**（图像存储为 MP4 视频，Parquet 仅存储状态）之间进行双向转换。

## 📋 功能说明

1.  **`robotwin_to_corobot.py`**: 
    - 将 RoboTwin 数据集转换为 CoRobot/LeRobot 标准格式。
    - 提取 Parquet 中的图像数据，生成 MP4 视频文件。
    - 生成新的 Parquet 文件（不含图像数据）和 metadata。
    - **特点**: 自动将列名添加 `_rgb` 后缀以符合 CoRobot 标准。

2.  **`corobot_to_robotwin.py`**:
    - 将 CoRobot 格式数据集还原为 RoboTwin 格式。
    - 从 MP4 视频中提取帧，转为 PNG 字节流。
    - 将图像数据重新封装回 Parquet 文件中。
    - **特点**: 自动移除 `_rgb` 后缀，还原为 struct 结构。

## 🛠️ 环境依赖

请确保安装以下 Python 库：

```bash
pip install pandas pyarrow numpy imageio imageio-ffmpeg opencv-python-headless tqdm
```

同时，系统需安装 `ffmpeg` (通常 `imageio-ffmpeg` 会自动处理，但建议系统也安装)。

## 🚀 使用方法

### 1. RoboTwin 转 CoRobot (Parquet -> Video)

```bash
python3 robotwin_to_corobot.py \
  --source /path/to/robotwin_dataset \
  --target /path/to/output_corobot_dataset
```

**示例**:
```bash
python3 robotwin_to_corobot.py \
  --source ./RoboTwin_dataset/aloha_adjustbottle_left_source \
  --target ./RoboTwin_converted_to_CoRobot
```

### 2. CoRobot 转 RoboTwin (Video -> Parquet)

```bash
python3 corobot_to_robotwin.py \
  --source /path/to/corobot_dataset \
  --target /path/to/output_robotwin_dataset
```

**示例**:
```bash
python3 corobot_to_robotwin.py \
  --source ./RoboTwin_converted_to_CoRobot \
  --target ./RoboTwin_restored
```

### 3. 运行测试

使用提供的测试脚本验证转换流程的正确性：

```bash
python3 test_conversion.py
```
该脚本会执行一次完整的 "源 -> CoRobot -> 还原" 流程，并检查文件结构和数据完整性。

## 📂 数据结构对比

| 特性 | RoboTwin (源) | CoRobot (目标) |
|------|---------------|----------------|
| **图像存储** | 内嵌于 Parquet (`struct<bytes, path>`) | 独立 MP4 视频文件 (`videos/`) |
| **Parquet内容** | 状态 + 图像二进制流 | 仅状态 (State/Action) |
| **图像列名** | `observation.images.cam_high` | `observation.images.cam_high_rgb` |
| **Metadata** | `dtype: struct` (隐式) | `dtype: video` |

## ⚠️ 注意事项

- **源数据完整性**: 提供的源数据中，部分 Parquet 文件（如 `episode_000001.parquet` 及之后）大小为 0 字节。脚本会跳过这些损坏的文件并打印错误信息，这是正常现象。
- **CoRobot 参考数据**: 原始的 `Cobot_Magic_pour_water_bottle` 文件夹中的 Parquet 文件是 Git LFS 指针，无法直接读取。本工具生成的 CoRobot 格式数据是真实的 Parquet 文件，可以直接被 LeRobot 等框架加载。
