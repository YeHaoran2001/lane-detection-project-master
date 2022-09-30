# A Toy Lane Detection Project with Opencv-python

### Haoran Ye
### Soochow University  
<br>


To detect straight lanes:
```bash
python run.py --path ./data/image/straight.jpg
```

To detect curved lanes:
```bash
python run.py --path ./data/image/curve.jpg --disable_erode
```

To detect for a video:
```bash
python run.py --path ./data/video/project_video.mp4 --is_video --intialTracbarVals 42 63 13 87
```

To detect with haze:
```bash
python run.py --path ./data/image/haze.jpg --dehaze --intialTracbarVals 40 63 13 87
```

To detect with shadow:
```bash
python run.py --path ./data/image/shadow.jpg --deshadow --intialTracbarVals 40 63 13 87 --disable_erode
```

To detect at night:
```bash
python run.py --path ./data/image/night.jpg --illumination --intialTracbarVals 41 63 13 87 --disable_erode
```
