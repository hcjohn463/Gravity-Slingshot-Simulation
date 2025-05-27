# 🖼️ 自定義圖片使用指南

## 📁 支援的圖片格式

程式會自動尋找以下檔案名稱的圖片：

### 🌙 月球圖片
- `moon.png`
- `moon.jpg` 
- `moon.jpeg`

### 🚀 太空船圖片
- `spacecraft.png`
- `spacecraft.jpg`
- `spacecraft.jpeg`
- `spaceship.png`

## 🎯 使用方法

### 1. 準備圖片檔案
將你的月球和太空船圖片放在與程式相同的資料夾中，並使用上述檔案名稱。

### 2. 圖片建議
- **格式**: PNG格式最佳（支援透明背景）
- **大小**: 建議 200x200 到 500x500 像素
- **背景**: 透明背景效果最好
- **內容**: 
  - 月球：圓形的月球圖片
  - 太空船：側視圖的太空船/火箭

### 3. 執行程式
```bash
python enhanced_slingshot.py
# 或
python realistic_slingshot.py
```

## 📊 程式行為

### ✅ 有圖片時
```
✅ Moon image loaded successfully
✅ Spacecraft image loaded successfully
```
- 使用你的自定義圖片
- 圖例顯示方形標記

### ℹ️ 沒有圖片時  
```
ℹ️ Using default moon circle (no moon image found)
ℹ️ Using default spacecraft dot (no spacecraft image found)
```
- 使用預設的圓圈和點
- 圖例顯示圓形標記

## 🎨 圖片範例建議

### 月球圖片
- 可以使用真實的月球照片
- 或卡通風格的月球圖案
- 建議有陰影效果的圓形

### 太空船圖片
- 火箭或太空船的側視圖
- 可以是NASA風格的真實太空船
- 或科幻風格的太空船設計

## 🔧 技術細節

- 圖片會自動轉換為RGBA格式
- 月球圖片縮放比例：0.06x（可調整）
- 太空船圖片縮放比例：0.05x（可調整）
- 支援透明背景
- 自動處理不同圖片格式

### 🎛️ 調整圖片大小

如果圖片太大或太小，可以修改程式碼中的縮放參數：

**月球圖片大小調整：**
```python
# 在 enhanced_slingshot.py 和 realistic_slingshot.py 中找到這行：
moon_img = OffsetImage(self.moon_image, zoom=0.06)  # 調小月球圖片

# 調整 zoom 值：
# - 0.03 = 很小
# - 0.06 = 適中（目前設定）
# - 0.1 = 較大
# - 0.15 = 很大
```

**太空船圖片大小調整：**
```python
# 找到這行：
spacecraft_img = OffsetImage(self.spacecraft_image, zoom=0.05)

# 調整 zoom 值：
# - 0.03 = 很小
# - 0.05 = 適中（目前設定）
# - 0.08 = 較大
# - 0.1 = 很大
```

## 💡 提示

1. **透明背景**: PNG格式可以有透明背景，看起來更自然
2. **檔案大小**: 不需要太大的圖片，程式會自動縮放
3. **測試**: 可以先用一個圖片測試，再添加另一個
4. **備份**: 如果圖片載入失敗，程式會自動使用預設圖形

## 🚀 開始使用

1. 找到你喜歡的月球和太空船圖片
2. 重新命名為指定的檔案名稱
3. 放在程式資料夾中
4. 執行程式享受自定義的重力彈弓動畫！

現在你可以用自己的圖片來個性化重力彈弓模擬了！🌙🚀 