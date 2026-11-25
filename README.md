# 滚动时基校正的事件体素化交付

两台合成事件读头沿不同扫描行时钟对同一表面取样。数据为本地合成。

- `data/capture.json` 保存采集原始内容。records 是交换记录，tensors 字段分开描述名称、类型、形状和值。整数索引从零开始，未指定闭合的区间均为左闭右开。
- `data/capacity.json` 保存制造端的容量、规模或分片条件。大尺寸是接口需考虑的使用规模。工作区计入新分配的解包、转换、归约、版本暂存及交接缓冲。调用方已有只读输入和已消费的输出不计入工作区，额外物化副本计入。

## 交换约定

事件数组是事件 ID 序号、纵坐标、横坐标、读头本地时标、极性。读头身份与上报批次附带；极性为 −1 或 +1。标定有效期按原始本地时标左闭右开。空间坐标应用每轴比例及偏置，校正时间为 `t + offset + row × line_delay`，`row` 使用原始坐标。实坐标落在某拥有盒内才有效，盒左右界由所用的标定版声明。

按连续坐标在相邻空间和时间格点之间各作线性权重分配，以体素原点和步长为单位。三个轴的权重相乘，出空间边界的份额记裁损，两个时间窗可各收到一部分。每个窗四个整时格，按 `[start,end)`。拥有盒所声明的读头提供覆盖证据，进度的本地时标是到该时刻之前所有事件已传完的承诺；转成全局时要考虑所有拥有行和插值支持。

## 交付代码

- `event_voxel/`：算子、标定版本、有界窗存储、持续输入和交接。
- `voxelizer.py`、`handoff.py`、`event_voxelizer.py`：常见导入名的兼容入口。
- `replay_capture.py`：回放 `data/capture.json` 的命令行入口。
- `tests/test_event_voxel.py`：权重、身份、拥有权、进度、修订冲突和活跃空间测试。

```sh
python -m pip install -r requirements.txt
python caseio.py data/capture.json
python -m unittest discover -s tests -v
python replay_capture.py data/capture.json --summary-only
```

隔离运行读取 JSON 的脚本时可使用：

```sh
python -I /workspace/replay_capture.py /workspace/data/capture.json --summary-only
```

## 体素口径

规范张量为 `voxels.shape = (2, T, H, W)`：

- 通道 0：负极性（−1）；
- 通道 1：正极性（+1）；
- `T = window_size`，样例中为 4；
- 需要时间在前的布局时使用 `delivery.voxels_tchw`，形状 `(T,2,H,W)`；
- `delivery.signed_voxels_thw` 为正通道减负通道。

全局格点坐标为：

```text
gy = (y_scale * raw_y + y_shift - origin_y) / step_y
gx = (x_scale * raw_x + x_shift - origin_x) / step_x
gt = (t_local + offset + raw_y * line_delay - origin_t) / time_step
```

对每个坐标取 floor/ceil 两端并作线性权重；空间、时间权重相乘。时间格点 `k` 属于全局窗 `floor(k / T)`、窗内下标 `k mod T`，因此一个事件最多进入相邻两个时间窗。精确落在格点上时不分配给零权重邻点，也不会为空邻点分配窗。

**边界分配损失**是本应分配到 `[0,H)×[0,W)` 之外空间格点的乘积权重之和；时间跨窗不是损失，因为时间轴不设全长边界。每个窗可读取：

- `boundary_loss`：归属到该窗的加权裁损总量；
- `boundary_loss_by_time`：该窗每个时间格点对应的裁损；
- `clipped_event_components`：被裁掉的非零角点数量。

恒等式为：

```text
窗群体素质量之和 + 已计量边界损失 = 可用事件单位权重
```

跨两个时间窗的损失按时间权重分别计入两个窗，因此所有窗合计仍满足上式。

## 身份与空间拥有权

去重键是真实身份 `(sensor, event_id)`，不是批次名或张量数值：

- 同一读头重发相同事件 ID：第二次起计入 `duplicates`，不重复累计；
- 不同读头使用相同数字 ID：属于不同身份，不会被身份去重；
- 投影坐标不属于该读头自己的拥有盒：计入 `owner_rejected/rejected_ids`，不进入张量；
- 两个读头即使投影出相同坐标/时间，也只按拥有盒判断哪个读头有效，绝不按体素值或坐标哈希去重。

## 覆盖、进度和可推理性

`coverage` 是标定拥有盒光栅化出的空间覆盖掩码 `(H,W)`，与事件权重大小无关。边界裁损不会把整个窗或其他有效区域改成未覆盖。

读取结果还提供：

- `effective_coverage` / `zero_coverage`：只有窗的进度承诺完整时，才等于 `coverage`；未到承诺时为空掩码，避免把“尚未保证无事件”误读成零跃迁；
- `coverage_status`：`0` 未覆盖，`1` 可按零跃迁使用，`-1` 进度未闭合；已交接修订冲突的交付中为 `-2`；
- `complete` / `ready` / `usable`：当前窗是否达到推理可用的全局进度；
- `horizons`：每个拥有读头换算后的排他性全局完成时界。

本地进度 `p` 承诺所有 `t_local < p` 的事件已送达。对拥有盒反投影得到的原始行集合 `R`，全局完成时界取最慢行：

```text
horizon_sensor = min_{r in R}(p + offset + r * line_delay)
```

窗 `w` 的半开物理末端为：

```text
end_w = origin_t + (w + 1) * window_size * time_step
```

只有每个在覆盖图中拥有像素的读头都满足 `horizon >= end_w`，该窗才能发布无事件结论。这里使用窗末端而不是最后一个格点时刻，正是为了把最后一格到右侧相邻格点之间的线性插值支持计入。

## 持续输入与修订交接

典型接入：

```python
from event_voxel import StreamingVoxelizer, VoxelConfig
from event_voxel.replay import load_exchange

records, tensors = load_exchange("data/capture.json")
stream = StreamingVoxelizer(
    VoxelConfig.from_mapping(records),
    records["calibrations"],
    active_calibration="v1",
)

for submission in records["submissions"]:
    stream.submit(submission, tensors)

    while True:
        delivery = stream.claim_next()       # 未就绪时返回 None
        if delivery is None:
            break
        infer(delivery.voxels, delivery.effective_coverage)
        stream.ack(delivery, drop_window=True)
```

也可以手动：

```python
stream.ingest_events("east", events_array, batch_id="segment-42")
stream.update_progress("east", 20)
stream.activate_calibration("v2")

view = stream.window(0)
view.voxels                 # 当前修订分布
view.coverage               # 空间拥有覆盖
view.boundary_loss          # 边界分配损失
view.usable                 # 当前是否可供推理

old = stream.handed_window(0)
old.conflict                # 标定/进度更正后变为 True
old.conflicts               # 修订记录
old.voxels                  # 已交接的原版本，只读、不被后续重算改写
```

语义保证：

1. 标定版本切换后，所有保留的原始事件按新投影/时标重算；未交接窗是“先从旧分布移除，再计入新分布”，不会保留两份。
2. 已交接的 `WindowDelivery` 张量只读，始终保留交接时版本。
3. 后续标定更正或进度回退会给原交付挂 `calibration_revision` / `progress_revision` 冲突标记；原交付变为 `usable=False`，但仍可查询和审计。
4. 当前修订在新的进度承诺闭合后重新变为可用；历史冲突标记不会被抹掉。
5. 进度值按最后上报值保存；更低的值是更正，不做单调取最大值。

## 活跃空间与容量

算子不按长时间轴分配数组。只有事件命中、显式查看或待交接的窗才拥有稠密张量；`max_active_windows` 限制活跃窗数量。达到限制时新一段输入会抛出 `ActiveWindowLimitError`，调用方应先领取/确认并释放旧窗，而不是静默分配全时域。

默认交接为零拷贝转移：稠密数组转为只读并由 `WindowDelivery` 持有，算子删除活跃窗状态；归档字典和冲突标记保留可查性。需要算子继续保留可变状态时使用：

```python
delivery = stream.claim_window(w, copy=True)
# 推理消费完成后释放归档副本：
stream.ack(delivery, drop_window=True)
```

`capacity.json` 的 `peak_scratch_bytes` 会用于推导保守分块大小；投影、floor/frac、权重和 `np.add.at` 归约均按 `chunk_size` 分块，避免一次性物化 210 万事件的多份临时数组。紧凑原始批次约为 `5 × int64/事件`，`stream.workspace_bytes()` 可查看稠密窗、保留事件批次和归档交付的字节数。
