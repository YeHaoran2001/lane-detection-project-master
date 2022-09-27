# 机器视觉课程实践1：车道线即时检测

### 姓名：叶皓然
### 学号：2027407057  
<br>

## 进行测试：
<br>

检测直线车道线
```bash
python run.py --path ./data/image/straight.jpg
```

检测曲线车道线
```bash
python run.py --path ./data/image/curve.jpg --disable_erode
```

对视频进行检测
```bash
python run.py --path ./data/video/project_video.mp4 --is_video --intialTracbarVals 42 63 13 87
```

检测有雾天的车道线
```bash
python run.py --path ./data/image/haze.jpg --dehaze --intialTracbarVals 40 63 13 87
```

检测有阴影的车道线
```bash
python run.py --path ./data/image/shadow.jpg --deshadow --intialTracbarVals 40 63 13 87 --disable_erode
```

检测夜间的车道线
```bash
python run.py --path ./data/image/night.jpg --illumination --intialTracbarVals 41 63 13 87 --disable_erode
```
