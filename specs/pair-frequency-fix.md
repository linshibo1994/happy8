# 数字对频率分析修复 Spec

## 目标

修复 `engine/happy8_analyzer.py` 中数字对频率分析的统计口径问题，保证结果以有效开奖期数为分母，并以快乐8全量 `C(80, 2)=3160` 个数字对为完整分析范围。

## 约束

- 不回滚或提交工作区内已有的无关改动。
- 保持现有 `PairFrequencyAnalyzer.analyze_pair_frequency` 调用方式不变。
- 无效开奖行只跳过并计数，不中断整次分析。
- 缓存键必须感知历史数据变化，避免相同目标期号和统计期数复用旧数据结果。

## 验收标准

- `extract_number_pairs` 示例使用合法的 20 个开奖号码。
- `actual_periods` 表示通过号码校验的有效统计期数。
- `percentage` 使用有效统计期数作为分母。
- `skipped_invalid_rows` 记录统计范围内被跳过的无效行数量。
- `total_pairs` 固定表示全量 3160 个数字对，未出现的数字对以 `count=0`、`percentage=0` 输出。
- 历史数据变化后，相同参数调用不会命中过期缓存。
