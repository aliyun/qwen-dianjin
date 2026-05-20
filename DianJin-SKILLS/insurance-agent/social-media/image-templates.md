# 小红书封面图生成 Prompt 模板

使用 `ImageGen` 工具生成图片时，根据内容风格选择对应 prompt 模板。

## 通用设计规范

| 参数 | 值 |
|------|-----|
| 尺寸 | `1024x1280`（4:5竖版） |
| 主色调 | 根据风格选择（见下方） |
| 文字 | 中文，简洁标题，不超过15字 |
| 风格 | 小红书调性：简约、精致、有质感 |

## 风格一：干货科普型

**配色**：深蓝(#1A3B6B) + 白色 + 浅金(#D4A574)

**Prompt 模板**：

```
A professional Chinese insurance knowledge infographic poster for Xiaohongshu (RED) social media.
Dark blue (#1A3B6B) and white color scheme with gold (#D4A574) accents.
Title in large bold Chinese font: "[标题，如：医保改革3大变化]"
Subtitle in smaller font: "[副标题，如：打工人必看]"
Clean layout with numbered list area in the middle, showing 3 key points.
Bottom section with a warm tagline: "[如：专业守护，让保障更清晰]"
Minimalist flat design, subtle geometric patterns in background.
Social media post aesthetic, eye-catching and shareable.
Icons: shield, document, medical cross.
High quality, no watermark, elegant Chinese typography, Xiaohongshu style.
```

## 风格二：热点关联型

**配色**：渐变橙红(#FF6B35) + 深灰(#2D3436) + 白色

**Prompt 模板**：

```
A trending topic style Chinese insurance awareness poster for Xiaohongshu (RED) social media.
Warm gradient background from orange (#FF6B35) to coral (#FF8C42).
Bold attention-grabbing title: "[标题，如：暴雨来了，你的车险够用吗]"
Modern dynamic layout with diagonal elements and card-style content blocks.
Key message highlighted in white card overlay: "[核心信息]"
Bottom area with engaging call-to-action text.
Contemporary social media post design, viral content aesthetic.
Trendy, bold, high-contrast, designed for maximum scroll-stopping impact.
Icons: trending arrow, alert symbol, umbrella.
High quality, no watermark, impactful Chinese typography, Xiaohongshu style.
```

## 风格三：真实案例型

**配色**：暖白(#FFF8F0) + 墨绿(#2D5F3E) + 玫瑰金(#B76E79)

**Prompt 模板**：

```
A warm and emotional Chinese insurance storytelling poster for Xiaohongshu (RED) social media.
Soft warm white (#FFF8F0) background with gentle green (#2D5F3E) text.
Rose gold (#B76E79) accent elements.
Heartwarming title: "[标题，如：一份保障，守护一个家]"
Soft watercolor-style illustration of a family or person in the center.
Gentle curved design elements, no sharp edges, rounded corners.
Quote-style text area for the story highlight with quotation marks.
Bottom with warm closing message: "[如：愿每个家庭都有安心的底气]"
Elegant, tender, emotional design style.
Soft light, warm tones, comfort and trust feeling.
Icons: heart, hands, home, family silhouette.
High quality, no watermark, beautiful Chinese calligraphy-style title, Xiaohongshu style.
```

## Prompt 填充指引

生成图片时，按以下步骤填充模板：

1. **确定风格**：根据第2步选定的风格，选择对应 prompt 模板
2. **填充标题**：从新闻中提炼不超过15字的核心标题
3. **填充副标题**：补充说明，不超过20字
4. **填充核心信息**：1句话总结关键价值点
5. **填充结语**：选择温暖有力的收尾语
6. **调用 ImageGen**：使用填充后的完整 prompt，尺寸设为 `1024x1280`

## 图片质量检查

生成图片后，确认以下要素：

- [ ] 整体配色协调，符合对应风格
- [ ] 画面干净精致，无杂乱元素
- [ ] 设计风格符合小红书调性（精致、有质感）
- [ ] 适合在手机上浏览（竖版4:5比例）
- [ ] 标题文字清晰醒目，能在缩略图中看清
