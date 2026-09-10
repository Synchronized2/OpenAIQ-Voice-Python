# OpenAIQ Voice Python

Windows 语音对话程序：FireRedVAD 实时端点检测、Paraformer-large 本地识别、OpenAI 兼容聊天节点、Edge TTS 朗读。

## 新电脑首次安装

使用 **Windows 10/11、64 位 Python 3.11 或 3.12**，无需复制旧电脑的虚拟环境。
主要原生依赖均提供 Python 3.12 的 Windows x64 wheel；仓库包含两个版本的 CI 测试。
模型约 891 MB，Python、PyTorch 等依赖还需要额外磁盘空间，建议预留至少 5 GB。

```powershell
git clone https://github.com/Synchronized2/OpenAIQ-Voice-Python.git
cd OpenAIQ-Voice-Python
python --version
.\setup.ps1
# 编辑自动生成的 config.py，填写 CHAT_BASE_URL 和 CHAT_API_KEY
.\run-ui.ps1
```

如果默认 `python` 不是所需版本，可指定解释器（替换成实际路径）：

```powershell
.\setup.ps1 -Python "C:\Python312\python.exe"
```

`-Python` 只用于创建新的 `.venv`，已有虚拟环境会继续使用，不会自动更换版本或删除。
想更换 Python 版本时，先关闭程序，将旧 `.venv` 改名备份，再运行上述命令。
若 PowerShell 阻止脚本执行，可在当前终端运行
`Set-ExecutionPolicy -Scope Process Bypass`，不会更改永久系统策略。

安装过程会：

- 将依赖安装在项目 `.venv` 中，pip 使用 `--no-cache-dir`，不主动清除你已有的全局缓存。
- 在不存在 `config.py` 时从 `config.example.py` 创建，绝不覆盖已有配置和 Key。
- 从 ModelScope 下载 `models-manifest.json` 列出的必要 ASR/VAD 文件到 `models/`。
- 对文件大小和 SHA-256 校验，跳过正确的文件；失败后可重新运行，不重下已验证的文件。
- 将下载临时文件放在项目 `.cache/`；下载失败不会替换现有模型。

Paraformer 固定为 `v2.0.5`；FireRedVAD 使用 `master` 并固定文件校验值。
如果上游文件发生变化，安装会明确报错，不会静默使用不同模型。
下载仅在安装时发生，启动不会重新下载。新电脑默认使用系统默认麦克风，可在界面更改。

```powershell
# 仅安装依赖；之后运行语音功能前必须下载模型
.\setup.ps1 -SkipModels
# 单独下载或校验模型
.\.venv\Scripts\python.exe .\download_models.py
.\.venv\Scripts\python.exe .\download_models.py --check
```

## 两台电脑同步

只同步源码，不同步虚拟环境、模型、录音、日志、缓存、`config.py` 或 `ui-settings.json`。
这些已加入 `.gitignore`；Key 在每台电脑分别填写，也可使用 `PCIE_API_KEY` 环境变量。
不要用 `git add -f config.py` 强行提交密钥。第三方 VAD 运行源码和桌宠图片属于运行必需资产，会提交。

每次开始修改前运行 `git pull --ff-only`。完成修改并测试后：

```powershell
git status
git add .
git diff --cached --stat
git commit -m "Describe your changes"
git push
```

另一台电脑运行 `git pull --ff-only` 获取更新；依赖或模型清单变化后重跑 `setup.ps1`。
配置模板更新时，对照 `config.example.py` 手动补充新配置项，保留自己的 Key。
若两边都产生了新提交，`--ff-only` 会停止，先合并解决冲突，不要强制推送覆盖对方。
本地仓库与 GitHub 的同步通过 Git 完成，GitHub 登录凭据与模型 API Key 无关。

## 配置

编辑 `config.py`（也可以在启动菜单选择模型）：

- `CHAT_API_KEY`：可选，留空时依次读取系统环境变量 `PCIE_API_KEY`、`OPENAI_API_KEY`
- `CHAT_MODEL`：启动时直接回车使用的默认模型，新安装默认为 `gpt-5.3-codex-spark`
- `WAKE_WORD_ENABLED`：图形界面是否在桌面常驻状态监听“悟空”等唤醒词
- `ASR_VAD_END_SILENCE`：检测到语音后，连续静音多久提交当前命令，默认 `0.7` 秒
- `ASR_HOTWORDS`：应用名称、唤醒词和 Agent 指令中的识别热词
- `TTS_BARGE_IN_ENABLED`：是否默认允许在朗读时说出下一问题来打断，当前默认开启
- `TTS_VOICE`：默认 Edge-TTS 音色，图形界面可从六个精选中文音色中切换
- `TTS_RATE`：默认语速；图形界面提供 -30% 到 +30% 调节
- `TTS_VOLUME`：默认播放音量，范围 `0.0` 到 `1.0`
- `UI_CORE_STYLE`：悬浮核心风格，可选 `PET`（孙悟空精灵）、`A`（赛博能量环）、
  `D`（金色角色光环）或 `A+D`
- `PET_SPRITESHEET`：`PET` 模式使用的 8×9 透明精灵图路径
- `COMPACT_PET_SIZE`：桌面宠物默认大小；也可直接在图形界面中滑动调节
- `UI_ANIMATION_FPS`：默认动画刷新率；也可直接在图形界面中滑动调节
- `PET_FRAME_INTERVAL_MS`：宠物动画每帧持续时间，默认 `150` 毫秒
- `WAKE_WORDS`：可识别的本地唤醒短语；关闭唤醒词后仍可使用 `Ctrl + Space`
- `WAKE_WORD_ALIASES`：常见的同音识别结果，例如“五空”“大胜”
- `WAKE_COOLDOWN_SECONDS`：成功唤醒后的冷却时间，防止同一段声音重复触发
- `WAKE_MIN_CHARS` / `WAKE_MAX_CHARS`：过滤过短噪声和包含唤醒词的过长背景对白
- `AGENT_ENABLED`：是否启用高置信度本地桌面操作
- `AGENT_ADJUST_STEP`：未指定数值时，音量和亮度每次调节的百分比
- `AGENT_APPLICATIONS`：自定义应用名称与 exe/快捷方式完整路径
- 启动菜单可选择 `gpt-5.6-luna`、`gpt-5.6-terra`、`gpt-5.6-sol` 或 `qwen3.7-plus`
- 天气问题会自动调用无需 Key 的 Open-Meteo；可在 `config.py` 设置 `WEATHER_DEFAULT_CITY`，也可以在问题中明确城市（例如“明天上海天气怎么样”）

Edge TTS 偶发返回空音频时会自动重试 3 次。

图形界面的设置面板提供“允许说话打断朗读”开关。开启时会在后台预热 ASR；TTS 真正
开始播放后，语音层依次显示“可直接插话”“已打断，请继续说完”“正在识别新问题”。
检测到人声会立即停止当前朗读和模型输出，录到句末后串行提交下一问题；打断监听完全结束
后才会恢复普通 ASR，避免两路录音争用同一模型。外放声音仍可能被麦克风当成人声，使用
该功能时建议佩戴耳机或使用带回声消除的音频设备。

插话结果使用独立任务保存，只能提交一次。停止语音、切换设备或输入新问题后，旧录音的
转写结果会被丢弃。Paraformer 已开始的 CPU 转写不能强行中止，但完成后不会执行旧指令。
LLM 和 Edge TTS 的网络等待支持取消，取消检查间隔为 50 毫秒；实际延迟还包括网络资源清理。
聊天连接超时 10 秒、读取超时 30 秒，不自动重试。同步天气查询仍受天气工具自身超时约束。

语音浮层下方只显示 TTS 当前正在播放的句子：新句开始时直接替换上一句，不再拼接模型的
全部流式输出；朗读结束或被打断后自动清空。完整聊天窗口仍保留完整回答和耗时信息。

GPT 回复会按句号、问号、感叹号等标点实时切分并送入后台 TTS 队列。第一句生成后即可
开始朗读，不需要等待整段回复完成；超过 `TTS_SEGMENT_MAX_CHARS` 仍无句末标点时会自动
按逗号或长度切分。短于 `TTS_SEGMENT_MIN_CHARS` 的句子会和后一句合并，避免音频过短导致
预合成追不上播放速度。

语音合成与播放使用两个独立线程：播放当前句子时，下一句会提前通过 Edge TTS 合成，
避免每个分片之间再次等待网络请求。

终端默认运行语音模式：说话后连续静音 `ASR_VAD_END_SILENCE` 秒会自动把最终文本发送给聊天模型，
回复流式生成并由 Edge TTS 播放；播放期间暂停麦克风，避免扬声器回声触发下一句。
ASR 模型保存在本项目的 `models/paraformer-large`，VAD 模型保存在 `models/FireRedVAD`。
FireRedVAD 检测到第一处语音端点后便立即提交，不会把较长静音之后的背景说话拼入当前命令。
新安装默认使用系统麦克风（`ASR_DEVICE = None`）；设备编号可通过 `--list-devices` 查看。
如果在 AI 生成或 TTS 阶段按 `Ctrl+C`，当前未完成的对话轮次会回滚，不会影响下一句。

```powershell
# 语音模式（默认）
.\run.ps1

# 跳过菜单，直接指定模型
.\run.ps1 --model gpt-5.6-sol

# 调整 VAD：连续静音 0.8 秒结束一句
.\run.ps1 --vad-end-silence 0.8

# 显示 FireRedVAD 概率与端点诊断
.\run.ps1 --debug-audio

# 临时追加识别热词
.\run.ps1 --hotwords "孙悟空 猴哥 悟空 大圣 QQ音乐 OpenAIQ 微信"

# 保留原有键盘输入模式
.\run.ps1 --text
```

## 安装和运行

```powershell
.\setup.ps1
.\run.ps1
```

## 图形界面

图形界面与终端版本共用相同的 ASR、LLM、TTS、天气和配置。安装依赖后运行：

```powershell
.\run-ui.ps1
```

图形界面分为三层：桌面常驻时只显示透明 AI 核心；按 `Ctrl + Space`，或说“孙悟空”、
“猴哥”、“悟空”、“大圣”、“齐天大圣”，会打开语音浮层并开始聆听；点击桌面核心或
语音浮层右上角的对话按钮，会打开完整聊天。

设置面板提供精选音色、音色试听、语速、音量、2D 形象、“桌宠大小”和“动画刷新率”控制，调整后立即生效并保存到 `ui-settings.json`，
下次启动会自动恢复。朗读、语音打断、唤醒开关、聊天模型和麦克风选择也会保存；设置文件
优先于 `config.py` 的默认值。切换麦克风仅切换音频输入，不重新加载模型。
语音波形使用实际麦克风电平，无输入时回落；隐藏窗口会停止动画定时器。改变刷新率不会
重载宠物精灵或重置当前动作，能量环按实际经过时间旋转。
悬浮核心支持鼠标左键拖动，单击核心才会打开完整聊天；拖动距离超过约 4 像素不会误触发打开。
悬浮核心下方会显示本地模型状态：首次初始化时为“出世中…”，FireRedVAD 和 Paraformer
加载成功后短暂显示“猴王出世”，随后回到“待命”。`PET` 模式会分别使用工作、挥手、
失败和待机动画行表达这些状态；缺少精灵图时会自动回退到 A+D 能量核心。
关闭完整聊天会回到桌面核心，彻底退出需使用托盘菜单中的“退出”。

### Live2D 形象

主窗口内容区顶部会持续显示 `assets/live2d/hiyori_pro_zh` 中的 Hiyori / 日和模型；切换到聊天记录、语音浮层或朗读状态时角色仍保持可见，监听、思考、执行、朗读和错误状态会驱动对应动作。桌面常驻小窗继续使用轻量的悟空精灵，以降低常驻渲染开销。

不要直接双击 `assets/live2d/live2d-viewer.html`，它是 Voice 内部使用的渲染页面，不是独立网页入口；请运行 `run-ui.ps1`，由程序提供本地资源服务并传入当前模型路径。

设置面板的“2D 形象”区域支持选择模型目录并递归扫描。将包含 `.model3.json`、`.model.json`、兼容 `index.json` 或 OpenAIQ `*.avatar.json` 的完整模型目录复制到任意位置，点击“选择目录”后扫描即可；程序只列出核心文件和纹理引用完整的模型。模型选择会保存到 `ui-settings.json`，下次启动自动恢复。

仓库还包含根据 `new_models/png` 素材制作的“婚纱雏田 / Wedding Hinata”2.5D 角色包。它可在同一个形象列表中扫描和切换，支持呼吸、摇摆、鼠标视差、眨眼、朗读口型和状态表情。该角色包采用 OpenAIQ 的 `*.avatar.json` 格式，不是 Cubism Editor 导出的 `.moc3`；平面图片无法自动恢复为原生 Live2D 网格和变形器工程。

Live2D 渲染使用 PySide6 WebEngine，`setup.ps1` 会通过 `requirements.txt` 安装 `PySide6-Addons`。若运行环境未安装 WebEngine，界面仍可启动，但会显示安装提示而不会影响语音功能。

Hiyori 是 Live2D 官方 Momose Hiyori PRO 示例模型，随 Mira 项目资源提供；请仅按模型作者和 Live2D 官方示例的许可范围进行个人使用，发布或商业使用前应单独确认模型授权。

完整聊天采用无侧栏、无气泡底板的自然排版，支持流式回复、文字输入、语音连续对话、模型
与输入设备选择、朗读开关、停止当前任务和新建对话。唤醒词识别完全在本机运行；可在设置
面板关闭并自动记住选择。终端的 `run.ps1` 仍读取 `config.py`。

GUI 使用统一运行状态管理 `BOOTING`、`STANDBY`、`READY`、`LISTENING`、`PROCESSING`、
`EXECUTING`、`SPEAKING` 和 `ERROR`，同一时间只允许 ASR、模型请求或本地 Agent 中的一个
工作单元运行。桌面和语音浮层中的悟空会随状态切换待机、倾听、思考、执行、说话和失败动作。
同音唤醒仅在 Paraformer 输出最终文字后进行一次拼音比较；桌面待机期间只持续运行轻量的 FireRedVAD，检测到一句完整语音后才调用 Paraformer。

## 本地桌面 Agent

文字和 ASR 识别结果会先经过严格的本地意图判断，明确的命令无需请求大模型即可执行：

- 设置、调高、调低或静音 Windows 主音量
- 设置或调节内置屏幕亮度（显示器需要支持 Windows WMI 亮度控制）
- 打开 Windows 设置、常用系统工具、开始菜单中的应用及自定义应用
- 播放/暂停、上一首、下一首、停止，锁屏和显示桌面

播放和暂停通过 Windows 媒体会话分别执行，不再使用同一个切换键。需要媒体应用支持
系统媒体会话；没有活动会话或应用拒绝操作时会明确告知。新依赖 `winsdk` 由 `setup.ps1` 安装。

例如：`音量调到 33%`、`帮我把音量提高到 60%`、`降低亮度`、`open qq music`、
`打开记事本`、`下一首`。带“帮我”的目标值表达会直接执行本地操作，不会先请求 LLM。
应用名称采用完整名称优先匹配，不会把 `QQ Music` 截断成 `QQ`。不能明确判断为本地命令的
内容仍按正常对话发送给模型。语音浮层识别到 `拜拜`、`再见`、`退出语音`、`结束对话`
或 `先这样` 时，会立即结束本轮语音交互并回到桌面常驻核心。

启动时先选择聊天模型，直接回车使用默认模型。默认进入麦克风语音模式，说话后连续静音会
自动发送，输入或识别到 `q` 可退出。使用 `--text` 可切换到原有键盘输入模式。

天气查询需要网络连接。若未提供城市且 `WEATHER_DEFAULT_CITY` 为空，助手会先询问所在城市；天气服务不可用时会明确说明，不会编造实时数据。

## 诊断日志

语音聊天默认向 GPT-5 系列兼容节点发送 `reasoning_effort=low`。这能在支持此参数的节点上
减少推理等待，但不保证固定延迟；复杂问题可把 `config.py` 的 `CHAT_REASONING_EFFORT`
设为 `auto`，恢复服务默认参数。其他模型不附加这个参数。若兼容节点拒绝该参数，也应设为 `auto`。

`CHAT_FIRST_TOKEN_TIMEOUT_SECONDS` 默认 30 秒，限制等待第一段正文的时间；流式心跳不会
延长它。`CHAT_TOTAL_TIMEOUT_SECONDS` 默认 90 秒，限制整次模型生成。超时会取消请求并丢弃
未完成的历史轮次。界面分别显示首字、模型生成和包含朗读的全程耗时。

真实延迟自测（使用当前 Key，会产生一次模型请求；不录音，默认不出声）：

```powershell
.\.venv\Scripts\python.exe diagnose_latency.py
# 对比节点默认推理参数
.\.venv\Scripts\python.exe diagnose_latency.py --effort auto
# 同时播放测试音频
.\.venv\Scripts\python.exe diagnose_latency.py --play
```

报告保存为 `logs/latency-*.json`，记录首字、首段音频就绪、MP3 解码时长等指标，不保存 Key
和对话内容。测试经过实际 ChatSession 与 EdgeSpeaker 管线，不加载 ASR 模型。

GUI 与终端启动后写入项目内 `logs/runtime.log`，最多保留当前文件及 3 个历史文件，
每个约 1 MB。记录录音开始、VAD 端点、ASR 转写耗时、模型首字、TTS 合成/播放及取消状态；
不写入 API Key、完整对话或录音。模型和朗读请求失败仍由界面/终端显示。

取消逻辑集中在 `cancellation.py`，插话结果生命周期集中在 `voice_session.py`。
