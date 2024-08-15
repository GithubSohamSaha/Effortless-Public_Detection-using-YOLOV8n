# Object-Detection-and-Prediction-using-faster-R-CNN-and-kalman-filter
# *Install the Ultralytics to use YoLoV8 for people detection*
pip install ultralytics
Collecting ultralytics
  Downloading ultralytics-8.2.50-py3-none-any.whl.metadata (41 kB)
[2K     [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m41.2/41.2 kB[0m [31m575.5 kB/s[0m eta [36m0:00:00[0m [36m0:00:01[0m
[?25hRequirement already satisfied: numpy<2.0.0,>=1.23.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (1.26.4)
Requirement already satisfied: matplotlib>=3.3.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (3.7.5)
Requirement already satisfied: opencv-python>=4.6.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (4.10.0.82)
Requirement already satisfied: pillow>=7.1.2 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (9.5.0)
Requirement already satisfied: pyyaml>=5.3.1 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (6.0.1)
Requirement already satisfied: requests>=2.23.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (2.32.3)
Requirement already satisfied: scipy>=1.4.1 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (1.11.4)
Requirement already satisfied: torch>=1.8.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (2.1.2)
Requirement already satisfied: torchvision>=0.9.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (0.16.2)
Requirement already satisfied: tqdm>=4.64.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (4.66.4)
Requirement already satisfied: psutil in /opt/conda/lib/python3.10/site-packages (from ultralytics) (5.9.3)
Requirement already satisfied: py-cpuinfo in /opt/conda/lib/python3.10/site-packages (from ultralytics) (9.0.0)
Requirement already satisfied: pandas>=1.1.4 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (2.2.1)
Requirement already satisfied: seaborn>=0.11.0 in /opt/conda/lib/python3.10/site-packages (from ultralytics) (0.12.2)
Collecting ultralytics-thop>=2.0.0 (from ultralytics)
  Downloading ultralytics_thop-2.0.0-py3-none-any.whl.metadata (8.5 kB)
Requirement already satisfied: contourpy>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (1.2.0)
Requirement already satisfied: cycler>=0.10 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (4.47.0)
Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (1.4.5)
Requirement already satisfied: packaging>=20.0 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (21.3)
Requirement already satisfied: pyparsing>=2.3.1 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (3.1.1)
Requirement already satisfied: python-dateutil>=2.7 in /opt/conda/lib/python3.10/site-packages (from matplotlib>=3.3.0->ultralytics) (2.9.0.post0)
Requirement already satisfied: pytz>=2020.1 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.1.4->ultralytics) (2023.3.post1)
Requirement already satisfied: tzdata>=2022.7 in /opt/conda/lib/python3.10/site-packages (from pandas>=1.1.4->ultralytics) (2023.4)
Requirement already satisfied: charset-normalizer<4,>=2 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (3.3.2)
Requirement already satisfied: idna<4,>=2.5 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (3.6)
Requirement already satisfied: urllib3<3,>=1.21.1 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (1.26.18)
Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/lib/python3.10/site-packages (from requests>=2.23.0->ultralytics) (2024.2.2)
Requirement already satisfied: filelock in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (3.13.1)
Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (4.9.0)
Requirement already satisfied: sympy in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (1.12.1)
Requirement already satisfied: networkx in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (3.2.1)
Requirement already satisfied: jinja2 in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (3.1.2)
Requirement already satisfied: fsspec in /opt/conda/lib/python3.10/site-packages (from torch>=1.8.0->ultralytics) (2024.3.1)
Requirement already satisfied: six>=1.5 in /opt/conda/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib>=3.3.0->ultralytics) (1.16.0)
Requirement already satisfied: MarkupSafe>=2.0 in /opt/conda/lib/python3.10/site-packages (from jinja2->torch>=1.8.0->ultralytics) (2.1.3)
Requirement already satisfied: mpmath<1.4.0,>=1.1.0 in /opt/conda/lib/python3.10/site-packages (from sympy->torch>=1.8.0->ultralytics) (1.3.0)
Downloading ultralytics-8.2.50-py3-none-any.whl (799 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m799.4/799.4 kB[0m [31m5.2 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
[?25hDownloading ultralytics_thop-2.0.0-py3-none-any.whl (25 kB)
Installing collected packages: ultralytics-thop, ultralytics
Successfully installed ultralytics-8.2.50 ultralytics-thop-2.0.0
Note: you may need to restart the kernel to use updated packages.

pip install -U ipywidgets
Requirement already satisfied: ipywidgets in /opt/conda/lib/python3.10/site-packages (7.7.1)
Collecting ipywidgets
  Downloading ipywidgets-8.1.3-py3-none-any.whl.metadata (2.4 kB)
Requirement already satisfied: comm>=0.1.3 in /opt/conda/lib/python3.10/site-packages (from ipywidgets) (0.2.1)
Requirement already satisfied: ipython>=6.1.0 in /opt/conda/lib/python3.10/site-packages (from ipywidgets) (8.20.0)
Requirement already satisfied: traitlets>=4.3.1 in /opt/conda/lib/python3.10/site-packages (from ipywidgets) (5.9.0)
Collecting widgetsnbextension~=4.0.11 (from ipywidgets)
  Downloading widgetsnbextension-4.0.11-py3-none-any.whl.metadata (1.6 kB)
Collecting jupyterlab-widgets~=3.0.11 (from ipywidgets)
  Downloading jupyterlab_widgets-3.0.11-py3-none-any.whl.metadata (4.1 kB)
Requirement already satisfied: decorator in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (5.1.1)
Requirement already satisfied: jedi>=0.16 in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.19.1)
Requirement already satisfied: matplotlib-inline in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.1.6)
Requirement already satisfied: prompt-toolkit<3.1.0,>=3.0.41 in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (3.0.42)
Requirement already satisfied: pygments>=2.4.0 in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (2.17.2)
Requirement already satisfied: stack-data in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (0.6.2)
Requirement already satisfied: exceptiongroup in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (1.2.0)
Requirement already satisfied: pexpect>4.3 in /opt/conda/lib/python3.10/site-packages (from ipython>=6.1.0->ipywidgets) (4.8.0)
Requirement already satisfied: parso<0.9.0,>=0.8.3 in /opt/conda/lib/python3.10/site-packages (from jedi>=0.16->ipython>=6.1.0->ipywidgets) (0.8.3)
Requirement already satisfied: ptyprocess>=0.5 in /opt/conda/lib/python3.10/site-packages (from pexpect>4.3->ipython>=6.1.0->ipywidgets) (0.7.0)
Requirement already satisfied: wcwidth in /opt/conda/lib/python3.10/site-packages (from prompt-toolkit<3.1.0,>=3.0.41->ipython>=6.1.0->ipywidgets) (0.2.13)
Requirement already satisfied: executing>=1.2.0 in /opt/conda/lib/python3.10/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.0.1)
Requirement already satisfied: asttokens>=2.1.0 in /opt/conda/lib/python3.10/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (2.4.1)
Requirement already satisfied: pure-eval in /opt/conda/lib/python3.10/site-packages (from stack-data->ipython>=6.1.0->ipywidgets) (0.2.2)
Requirement already satisfied: six>=1.12.0 in /opt/conda/lib/python3.10/site-packages (from asttokens>=2.1.0->stack-data->ipython>=6.1.0->ipywidgets) (1.16.0)
Downloading ipywidgets-8.1.3-py3-none-any.whl (139 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m139.4/139.4 kB[0m [31m1.2 MB/s[0m eta [36m0:00:00[0ma [36m0:00:01[0m
[?25hDownloading jupyterlab_widgets-3.0.11-py3-none-any.whl (214 kB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m214.4/214.4 kB[0m [31m3.4 MB/s[0m eta [36m0:00:00[0mta [36m0:00:01[0m
[?25hDownloading widgetsnbextension-4.0.11-py3-none-any.whl (2.3 MB)
[2K   [90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━[0m [32m2.3/2.3 MB[0m [31m17.8 MB/s[0m eta [36m0:00:00[0m00:01[0m00:01[0m
[?25hInstalling collected packages: widgetsnbextension, jupyterlab-widgets, ipywidgets
  Attempting uninstall: widgetsnbextension
    Found existing installation: widgetsnbextension 3.6.6
    Uninstalling widgetsnbextension-3.6.6:
      Successfully uninstalled widgetsnbextension-3.6.6
  Attempting uninstall: jupyterlab-widgets
    Found existing installation: jupyterlab-widgets 3.0.9
    Uninstalling jupyterlab-widgets-3.0.9:
      Successfully uninstalled jupyterlab-widgets-3.0.9
  Attempting uninstall: ipywidgets
    Found existing installation: ipywidgets 7.7.1
    Uninstalling ipywidgets-7.7.1:
      Successfully uninstalled ipywidgets-7.7.1
Successfully installed ipywidgets-8.1.3 jupyterlab-widgets-3.0.11 widgetsnbextension-4.0.11
Note: you may need to restart the kernel to use updated packages.

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch

# Use the model
model.train(data="/kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/data.yaml", epochs=30) 
Ultralytics YOLOv8.2.50 🚀 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla T4, 15095MiB)
[34m[1mengine/trainer: [0mtask=detect, mode=train, model=yolov8n.yaml, data=/kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/data.yaml, epochs=30, time=None, patience=100, batch=16, imgsz=640, save=True, save_period=-1, cache=False, device=None, workers=8, project=None, name=train3, exist_ok=False, pretrained=True, optimizer=auto, verbose=True, seed=0, deterministic=True, single_cls=False, rect=False, cos_lr=False, close_mosaic=10, resume=False, amp=True, fraction=1.0, profile=False, freeze=None, multi_scale=False, overlap_mask=True, mask_ratio=4, dropout=0.0, val=True, split=val, save_json=False, save_hybrid=False, conf=None, iou=0.7, max_det=300, half=False, dnn=False, plots=True, source=None, vid_stride=1, stream_buffer=False, visualize=False, augment=False, agnostic_nms=False, classes=None, retina_masks=False, embed=None, show=False, save_frames=False, save_txt=False, save_conf=False, save_crop=False, show_labels=True, show_conf=True, show_boxes=True, line_width=None, format=torchscript, keras=False, optimize=False, int8=False, dynamic=False, simplify=False, opset=None, workspace=4, nms=False, lr0=0.01, lrf=0.01, momentum=0.937, weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8, warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, pose=12.0, kobj=1.0, label_smoothing=0.0, nbs=64, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5, bgr=0.0, mosaic=1.0, mixup=0.0, copy_paste=0.0, auto_augment=randaugment, erasing=0.4, crop_fraction=1.0, cfg=None, tracker=botsort.yaml, save_dir=runs/detect/train3
Overriding model.yaml nc=80 with nc=1

                   from  n    params  module                                       arguments                     
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]                 
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]                
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]             
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]                
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]             
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]           
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]              
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]           
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]                  
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]                
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]                 
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]                 
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]           
YOLOv8n summary: 225 layers, 3011043 parameters, 3011027 gradients, 8.2 GFLOPs

[34m[1mTensorBoard: [0mStart with 'tensorboard --logdir runs/detect/train3', view at http://localhost:6006/
[34m[1mwandb[0m: Currently logged in as: [33msahasoham807[0m ([33msahasoham807-University of Calcutta[0m). Use [1m`wandb login --relogin`[0m to force relogin
Freezing layer 'model.22.dfl.conv.weight'
[34m[1mAMP: [0mrunning Automatic Mixed Precision (AMP) checks with YOLOv8n...
[34m[1mAMP: [0mchecks passed ✅
[34m[1mtrain: [0mScanning /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/labels... 15210 images, 1917 backgrounds, 0 corrupt: 100%|██████████| 15210/15210 [00:33<00:00, 452.78it/s][34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/000066_jpg.rf.7af12755c989607af4ff8faf8be1ed1b.jpg: 16 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2007_002387_jpg.rf.b185b9383eb87d476665f67d98e93996.jpg: 13 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2007_002845_jpg.rf.f86d7a1b7e9851bf8201be8dcaf3c8ec.jpg: 23 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2007_003226_jpg.rf.f1f5326d95fecd08c5a9dfdad9259e28.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2008_003526_jpg.rf.4a836fd57d560fc005eab941399b6500.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2008_003598_jpg.rf.df9555297acad95b8d1833b8cb78e7fb.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2008_003608_jpg.rf.11cb8904ac10f9f5fbaba7b3da1a5364.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2008_006361_jpg.rf.d7c25efb2ad609216a8fcff048099734.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2008_006517_jpg.rf.c43b88938c92123b015b583f2daabb2a.jpg: 23 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2009_002406_jpg.rf.eb6bdefd90894fd5853f7d886c34785c.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2009_002869_jpg.rf.2ba7834e2064f7d4355e7e88ee6db9d4.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2010_002363_jpg.rf.8442c12f3e10e557fc61850f3224e776.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2010_002498_jpg.rf.dcadcfb483196d8be7922fb603fcd478.jpg: 1 duplicate labels removed
[34m[1mtrain: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train/images/2010_004094_jpg.rf.fb5ac465bfa26e4053cac9c01b7c37a8.jpg: 1 duplicate labels removed

[34m[1mtrain: [0mWARNING ⚠️ Cache directory /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/train is not writeable, cache not saved.
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 428, len(boxes) = 99059. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
[34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
/opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
[34m[1mval: [0mScanning /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/valid/labels... 1431 images, 61 backgrounds, 0 corrupt: 100%|██████████| 1431/1431 [00:03<00:00, 386.34it/s][34m[1mval: [0mWARNING ⚠️ /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/valid/images/GX010023_frame_00025_right_jpg.rf.3255febc597d78d5d6e4bde455ba7b2a.jpg: 23 duplicate labels removed
[34m[1mval: [0mWARNING ⚠️ Cache directory /kaggle/input/public-detection-dataset-for-yolov8/People Detection.v8i.yolov8/valid is not writeable, cache not saved.
WARNING ⚠️ Box and segment counts should be equal, but got len(segments) = 61, len(boxes) = 10660. To resolve this only boxes will be used and all segments will be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.

Plotting labels to runs/detect/train3/labels.jpg... 
[34m[1moptimizer:[0m 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
[34m[1moptimizer:[0m AdamW(lr=0.002, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)
[34m[1mTensorBoard: [0mmodel graph visualization added ✅
Image sizes 640 train, 640 val
Using 2 dataloader workers
Logging results to [1mruns/detect/train3[0m
Starting training for 30 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/30      4.55G      3.017      2.989      3.076         66        640: 100%|██████████| 951/951 [03:28<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.12it/s]
                   all       1431      10660      0.218      0.148       0.11     0.0407

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/30      4.09G      2.165      2.221      2.071        143        640: 100%|██████████| 951/951 [03:26<00:00,  4.62it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.22it/s]
                   all       1431      10660      0.412      0.281      0.261      0.101

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/30      4.01G      1.937       1.96      1.841        145        640: 100%|██████████| 951/951 [03:23<00:00,  4.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.51it/s]
                   all       1431      10660      0.535      0.329      0.334      0.147

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/30      4.75G      1.821       1.81      1.733        109        640: 100%|██████████| 951/951 [03:23<00:00,  4.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.21it/s]
                   all       1431      10660      0.581       0.37      0.385      0.176

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/30      3.81G      1.733      1.701      1.662         75        640: 100%|██████████| 951/951 [03:22<00:00,  4.69it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.36it/s]
                   all       1431      10660      0.638      0.409      0.442      0.206

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/30      3.77G      1.674      1.612      1.614        104        640: 100%|██████████| 951/951 [03:24<00:00,  4.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.28it/s]
                   all       1431      10660      0.653      0.416       0.45      0.226

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/30      3.53G      1.617      1.544      1.574         83        640: 100%|██████████| 951/951 [03:25<00:00,  4.63it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.46it/s]
                   all       1431      10660      0.669      0.432      0.479      0.242

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/30      4.49G      1.589      1.508      1.554         99        640: 100%|██████████| 951/951 [03:28<00:00,  4.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.49it/s]
                   all       1431      10660      0.679      0.436      0.488       0.25

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/30      3.64G      1.554      1.468      1.528         96        640: 100%|██████████| 951/951 [03:28<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.26it/s]
                   all       1431      10660      0.704      0.443      0.506      0.262

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/30      4.07G      1.526      1.423      1.505         78        640: 100%|██████████| 951/951 [03:27<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.56it/s]
                   all       1431      10660      0.678      0.469      0.517      0.272

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/30      3.86G      1.509      1.397      1.486        152        640: 100%|██████████| 951/951 [03:28<00:00,  4.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.17it/s]
                   all       1431      10660        0.7      0.478      0.537      0.284

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/30      3.57G      1.492      1.381      1.477         91        640: 100%|██████████| 951/951 [03:24<00:00,  4.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.25it/s]
                   all       1431      10660      0.717      0.473      0.536      0.289

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      13/30      3.22G      1.463      1.346      1.456        101        640: 100%|██████████| 951/951 [03:24<00:00,  4.65it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.68it/s]
                   all       1431      10660      0.722      0.476      0.539      0.288

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      14/30       3.9G      1.442       1.32       1.44         78        640: 100%|██████████| 951/951 [03:24<00:00,  4.64it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.35it/s]
                   all       1431      10660      0.721      0.488      0.553      0.298

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      15/30      4.32G      1.436      1.305      1.431        185        640: 100%|██████████| 951/951 [03:23<00:00,  4.67it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.61it/s]
                   all       1431      10660      0.725      0.489      0.557      0.303

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      16/30      3.64G      1.418      1.282      1.419        107        640: 100%|██████████| 951/951 [03:24<00:00,  4.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.31it/s]
                   all       1431      10660      0.722      0.495      0.564      0.311

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      17/30      3.61G      1.404      1.275      1.412         92        640: 100%|██████████| 951/951 [03:23<00:00,  4.66it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.66it/s]
                   all       1431      10660      0.724      0.498      0.569      0.315

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      18/30      3.09G      1.391      1.251      1.402         77        640: 100%|██████████| 951/951 [03:23<00:00,  4.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.49it/s]
                   all       1431      10660      0.732      0.508       0.58      0.321

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      19/30      3.49G      1.381      1.239      1.391        154        640: 100%|██████████| 951/951 [03:23<00:00,  4.68it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.62it/s]
                   all       1431      10660      0.728      0.511      0.581      0.324

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      20/30      3.34G      1.367      1.228      1.387        102        640: 100%|██████████| 951/951 [03:21<00:00,  4.72it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.70it/s]                   all       1431      10660       0.74      0.511      0.581      0.324

Closing dataloader mosaic
[34m[1malbumentations: [0mBlur(p=0.01, blur_limit=(3, 7)), MedianBlur(p=0.01, blur_limit=(3, 7)), ToGray(p=0.01), CLAHE(p=0.01, clip_limit=(1, 4.0), tile_grid_size=(8, 8))
/opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()
/opt/conda/lib/python3.10/multiprocessing/popen_fork.py:66: RuntimeWarning: os.fork() was called. os.fork() is incompatible with multithreaded code, and JAX is multithreaded, so this will likely lead to a deadlock.
  self.pid = os.fork()

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      21/30      3.35G      1.336      1.181      1.387        101        640: 100%|██████████| 951/951 [03:16<00:00,  4.83it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.38it/s]
                   all       1431      10660      0.747      0.518      0.594       0.33

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      22/30      3.47G       1.32      1.156      1.374        124        640: 100%|██████████| 951/951 [03:13<00:00,  4.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.29it/s]
                   all       1431      10660      0.738      0.519      0.594      0.336

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      23/30      3.19G       1.31      1.143      1.371         75        640: 100%|██████████| 951/951 [03:12<00:00,  4.93it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.52it/s]
                   all       1431      10660      0.748      0.528      0.607      0.341

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      24/30      3.19G      1.298      1.123      1.361        104        640: 100%|██████████| 951/951 [03:12<00:00,  4.93it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.44it/s]
                   all       1431      10660      0.741       0.53      0.605      0.342

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      25/30      3.33G      1.287       1.11      1.353         71        640: 100%|██████████| 951/951 [03:14<00:00,  4.89it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.67it/s]
                   all       1431      10660      0.749      0.529      0.606      0.346

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      26/30      3.32G      1.269      1.097      1.344         37        640: 100%|██████████| 951/951 [03:12<00:00,  4.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.30it/s]
                   all       1431      10660      0.756      0.529      0.611      0.349

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      27/30      3.23G      1.262      1.084      1.335        120        640: 100%|██████████| 951/951 [03:13<00:00,  4.92it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.60it/s]
                   all       1431      10660      0.752      0.536      0.612      0.351

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      28/30      3.17G      1.248      1.068      1.327         47        640: 100%|██████████| 951/951 [03:12<00:00,  4.94it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.61it/s]
                   all       1431      10660      0.746      0.544      0.619      0.354

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      29/30      3.74G       1.24      1.058      1.322         74        640: 100%|██████████| 951/951 [03:12<00:00,  4.95it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:10<00:00,  4.35it/s]
                   all       1431      10660      0.749      0.541      0.618      0.355

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      30/30      3.05G      1.234      1.052      1.318         62        640: 100%|██████████| 951/951 [03:13<00:00,  4.93it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:09<00:00,  4.78it/s]
                   all       1431      10660      0.751      0.543       0.62      0.358

30 epochs completed in 1.777 hours.
Optimizer stripped from runs/detect/train3/weights/last.pt, 6.2MB
Optimizer stripped from runs/detect/train3/weights/best.pt, 6.2MB

Validating runs/detect/train3/weights/best.pt...
Ultralytics YOLOv8.2.50 🚀 Python-3.10.13 torch-2.1.2 CUDA:0 (Tesla T4, 15095MiB)
YOLOv8n summary (fused): 168 layers, 3005843 parameters, 0 gradients, 8.1 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 45/45 [00:17<00:00,  2.58it/s]
                   all       1431      10660      0.752      0.543       0.62      0.358
Speed: 0.2ms preprocess, 1.8ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to [1mruns/detect/train3[0m
ultralytics.utils.metrics.DetMetrics object with attributes:

ap_class_index: array([0])
box: ultralytics.utils.metrics.Metric object
confusion_matrix: <ultralytics.utils.metrics.ConfusionMatrix object at 0x7afb6c0bf9a0>
curves: ['Precision-Recall(B)', 'F1-Confidence(B)', 'Precision-Confidence(B)', 'Recall-Confidence(B)']
curves_results: [[array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[          1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,     0.99808,     0.99808,     0.99808,     0.99808,     0.99808,     0.99808,     0.99808,     0.99808,
            0.99808,     0.99808,     0.99808,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,      0.9971,     0.99609,     0.99609,     0.99609,     0.99609,
            0.99609,     0.99609,     0.99609,      0.9951,      0.9951,      0.9951,      0.9951,      0.9951,     0.99452,     0.99452,     0.99452,     0.99452,     0.99452,     0.99452,     0.99452,     0.99452,     0.99355,     0.99355,     0.99262,     0.99262,     0.99176,     0.99176,     0.99157,
            0.99157,     0.99157,     0.99157,     0.99157,     0.99157,     0.99157,     0.99157,     0.99157,     0.98988,     0.98911,     0.98911,     0.98911,     0.98911,     0.98911,     0.98911,     0.98911,     0.98911,     0.98911,     0.98911,     0.98836,      0.9868,     0.98637,     0.98637,
            0.98637,     0.98492,     0.98348,     0.98256,     0.98256,     0.98256,     0.98256,     0.98186,     0.98186,     0.98186,     0.98186,     0.98186,     0.98129,     0.98076,     0.98034,     0.98034,     0.97982,     0.97928,     0.97798,     0.97723,     0.97723,     0.97723,     0.97723,
            0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97723,     0.97709,     0.97709,     0.97709,     0.97698,     0.97698,     0.97698,     0.97698,     0.97673,     0.97673,      0.9766,      0.9766,      0.9766,
            0.97561,      0.9755,      0.9755,      0.9755,     0.97456,     0.97366,     0.97341,     0.97341,     0.97309,     0.97279,     0.97241,     0.97205,      0.9712,     0.97106,     0.97106,     0.97106,     0.97063,     0.97063,     0.97063,     0.97063,     0.97063,     0.97063,     0.97022,
            0.96994,     0.96913,     0.96839,     0.96827,     0.96827,     0.96791,     0.96719,     0.96616,     0.96616,     0.96616,     0.96616,     0.96616,     0.96587,     0.96538,     0.96538,     0.96538,     0.96538,     0.96538,     0.96527,     0.96527,     0.96503,     0.96477,     0.96382,
            0.96382,     0.96359,      0.9626,     0.96246,     0.96246,     0.96182,     0.96115,     0.96021,     0.96021,     0.96021,     0.96021,     0.96021,     0.96021,     0.95964,     0.95867,     0.95771,     0.95743,     0.95732,     0.95701,     0.95625,     0.95625,     0.95486,     0.95486,
            0.95486,     0.95486,     0.95458,     0.95369,     0.95352,     0.95332,     0.95278,     0.95229,     0.95211,     0.95204,     0.95204,     0.95123,     0.95123,     0.95123,     0.95073,      0.9504,      0.9504,      0.9504,      0.9504,      0.9504,      0.9504,      0.9504,      0.9504,
             0.9504,      0.9504,     0.95016,        0.95,        0.95,        0.95,     0.94758,      0.9471,      0.9466,      0.9466,     0.94641,     0.94564,     0.94526,     0.94526,     0.94516,     0.94516,     0.94516,     0.94443,     0.94435,     0.94435,     0.94435,     0.94427,      0.9438,
            0.94363,     0.94356,      0.9434,     0.94306,     0.94262,     0.94224,     0.94224,     0.94202,     0.94202,      0.9419,     0.94083,     0.94062,      0.9406,      0.9406,      0.9406,      0.9406,      0.9406,      0.9406,      0.9406,      0.9406,     0.94048,     0.94041,      0.9401,
            0.93974,     0.93968,     0.93949,     0.93949,     0.93949,     0.93949,     0.93949,     0.93949,     0.93949,     0.93869,     0.93826,     0.93817,     0.93817,     0.93815,     0.93796,     0.93657,     0.93602,     0.93602,     0.93602,     0.93602,     0.93542,     0.93465,     0.93381,
            0.93259,     0.93189,     0.93072,     0.93051,     0.93027,     0.93005,     0.92944,     0.92853,     0.92853,     0.92853,     0.92759,     0.92759,     0.92718,     0.92641,     0.92528,     0.92527,      0.9252,      0.9252,      0.9252,      0.9252,     0.92503,     0.92493,     0.92493,
            0.92493,     0.92464,     0.92428,     0.92428,     0.92422,     0.92414,     0.92361,     0.92281,     0.92274,     0.92259,     0.92235,     0.92164,     0.92126,      0.9212,      0.9212,      0.9212,     0.92023,     0.91944,     0.91846,     0.91824,     0.91777,     0.91616,     0.91561,
            0.91542,     0.91518,     0.91483,     0.91431,     0.91387,     0.91378,     0.91369,     0.91369,     0.91335,       0.913,     0.91248,       0.912,     0.91156,     0.91153,     0.91153,     0.91152,      0.9102,      0.9102,     0.90999,     0.90989,     0.90923,     0.90875,     0.90816,
            0.90749,     0.90718,     0.90696,     0.90696,     0.90647,     0.90609,     0.90489,     0.90489,     0.90461,     0.90434,     0.90346,     0.90328,     0.90224,     0.90136,      0.9009,     0.90023,     0.90012,     0.89903,      0.8987,     0.89834,     0.89826,     0.89792,     0.89744,
            0.89724,     0.89692,     0.89681,     0.89652,     0.89615,     0.89603,     0.89539,     0.89502,     0.89454,     0.89368,     0.89294,      0.8919,     0.89144,     0.89076,      0.8898,     0.88969,     0.88953,     0.88876,     0.88876,     0.88863,     0.88863,     0.88781,     0.88681,
            0.88602,     0.88513,     0.88513,     0.88454,     0.88347,      0.8833,      0.8831,      0.8824,     0.88189,     0.88071,     0.88067,     0.88034,     0.88024,     0.87919,     0.87748,     0.87616,     0.87493,     0.87493,     0.87417,     0.87253,     0.87109,     0.86913,     0.86782,
            0.86703,      0.8656,     0.86506,     0.86426,     0.86257,     0.86143,     0.86068,     0.85983,     0.85884,     0.85822,     0.85775,     0.85639,     0.85624,     0.85528,     0.85439,     0.85139,     0.85021,     0.84918,     0.84652,     0.84457,      0.8436,     0.84261,     0.84082,
            0.83955,      0.8385,     0.83797,     0.83713,     0.83701,     0.83483,     0.83373,     0.83238,     0.83009,     0.82902,     0.82866,     0.82742,     0.82598,       0.825,     0.82414,     0.82311,     0.82138,     0.82118,     0.81997,     0.81927,     0.81701,     0.81562,     0.81429,
            0.81227,     0.80979,     0.80862,     0.80692,     0.80526,     0.80428,     0.80264,     0.80161,     0.80041,     0.79977,     0.79777,     0.79656,     0.79428,     0.79219,     0.79057,     0.78927,     0.78731,     0.78572,     0.78388,     0.78254,     0.78157,     0.77747,     0.77605,
            0.77509,     0.77426,      0.7727,     0.77102,     0.76875,     0.76737,      0.7661,      0.7643,     0.76271,      0.7607,     0.75794,     0.75511,     0.75405,     0.75273,     0.75084,     0.74919,     0.74878,     0.74648,     0.74253,     0.74082,     0.73934,     0.73787,     0.73595,
            0.73472,     0.73332,     0.73232,     0.72872,      0.7264,     0.72311,     0.72094,     0.71935,     0.71898,     0.71516,     0.71327,     0.71197,     0.71074,      0.7095,     0.70751,     0.70549,     0.70313,     0.70035,     0.69803,     0.69547,     0.69379,     0.69174,      0.6894,
            0.68828,     0.68608,      0.6848,     0.68192,     0.67968,     0.67632,     0.67355,     0.67095,     0.66874,     0.66514,     0.66239,     0.65995,     0.65833,     0.65446,     0.65228,     0.65102,     0.64907,     0.64555,     0.64335,     0.63927,     0.63659,     0.63372,     0.63017,
            0.62803,     0.62477,     0.62347,     0.62205,       0.619,     0.61567,     0.61353,     0.61241,     0.61053,     0.60889,     0.60806,     0.60591,     0.60468,     0.60237,     0.59956,     0.59841,     0.59461,     0.59126,      0.5878,     0.58679,     0.58358,     0.58059,     0.57842,
            0.57695,     0.57343,     0.57202,     0.56978,      0.5675,     0.56668,     0.56493,     0.56201,     0.55901,     0.55751,     0.55551,     0.55256,     0.55071,     0.54897,     0.54654,     0.54383,      0.5415,     0.53704,     0.53315,     0.53156,     0.52941,     0.52651,     0.52406,
             0.5229,     0.51971,     0.51718,     0.51488,     0.51268,      0.5102,      0.5068,     0.50286,     0.49993,     0.49538,     0.49287,     0.49035,     0.48811,     0.48674,     0.48599,     0.48244,     0.48046,      0.4796,     0.47773,     0.47531,     0.47146,     0.46991,     0.46787,
            0.46614,      0.4638,     0.46324,     0.45928,     0.45623,     0.45349,     0.45221,     0.44996,     0.44767,     0.44395,     0.44025,     0.43656,     0.43309,     0.43104,     0.42772,     0.42432,     0.42067,     0.41728,     0.41499,     0.41257,     0.40978,     0.40717,     0.40512,
            0.40158,     0.40024,     0.39549,     0.39396,     0.39043,     0.38592,      0.3825,     0.38097,     0.37711,      0.3742,     0.37223,     0.36903,     0.36627,     0.36043,     0.35667,     0.35089,     0.34832,      0.3465,     0.34148,     0.33684,     0.33401,     0.33108,     0.32551,
            0.32272,     0.32062,     0.31781,     0.31304,     0.30942,     0.30532,     0.30042,     0.29666,     0.29143,     0.28843,     0.28459,     0.27776,     0.27613,     0.27127,     0.26944,     0.26297,     0.25914,     0.25686,     0.25215,     0.24866,     0.24403,     0.24056,     0.23642,
            0.23279,     0.22939,      0.2257,     0.22375,     0.21858,     0.21559,     0.21093,     0.20726,      0.2034,      0.2008,     0.19848,     0.19457,      0.1862,     0.18101,     0.17849,     0.17568,     0.17056,     0.16549,     0.16093,     0.15494,      0.1525,     0.14853,     0.14322,
              0.137,     0.13458,     0.13017,     0.12832,     0.12528,     0.11783,     0.11394,     0.11099,     0.10707,     0.10261,     0.10062,    0.095987,    0.093408,    0.091172,    0.084467,    0.081235,    0.078924,    0.073973,    0.070592,    0.067506,    0.061865,    0.061584,    0.061303,
           0.061021,     0.06074,    0.060459,    0.060178,    0.059897,    0.059615,    0.059334,    0.059053,    0.058772,    0.058491,    0.058209,    0.057928,    0.057647,    0.057366,    0.057085,    0.056803,    0.056522,    0.056241,     0.05596,    0.055679,    0.055397,    0.055116,    0.054835,
           0.054554,    0.054273,    0.053991,     0.05371,    0.053429,    0.053148,    0.052867,    0.052585,    0.052304,    0.052023,    0.051742,     0.05146,    0.051179,    0.050898,    0.050617,    0.050336,    0.050054,    0.049773,    0.049492,    0.049211,     0.04893,    0.048648,    0.048367,
           0.048086,    0.047805,    0.047524,    0.047242,    0.046961,     0.04668,    0.046399,    0.046118,    0.045836,    0.045555,    0.045274,    0.044993,    0.044712,     0.04443,    0.044149,    0.043868,    0.043587,    0.043306,    0.043024,    0.042743,    0.042462,    0.042181,      0.0419,
           0.041618,    0.041337,    0.041056,    0.040775,    0.040494,    0.040212,    0.039931,     0.03965,    0.039369,    0.039087,    0.038806,    0.038525,    0.038244,    0.037963,    0.037681,      0.0374,    0.037119,    0.036838,    0.036557,    0.036275,    0.035994,    0.035713,    0.035432,
           0.035151,    0.034869,    0.034588,    0.034307,    0.034026,    0.033745,    0.033463,    0.033182,    0.032901,     0.03262,    0.032339,    0.032057,    0.031776,    0.031495,    0.031214,    0.030933,    0.030651,     0.03037,    0.030089,    0.029808,    0.029527,    0.029245,    0.028964,
           0.028683,    0.028402,     0.02812,    0.027839,    0.027558,    0.027277,    0.026996,    0.026714,    0.026433,    0.026152,    0.025871,     0.02559,    0.025308,    0.025027,    0.024746,    0.024465,    0.024184,    0.023902,    0.023621,     0.02334,    0.023059,    0.022778,    0.022496,
           0.022215,    0.021934,    0.021653,    0.021372,     0.02109,    0.020809,    0.020528,    0.020247,    0.019966,    0.019684,    0.019403,    0.019122,    0.018841,     0.01856,    0.018278,    0.017997,    0.017716,    0.017435,    0.017153,    0.016872,    0.016591,     0.01631,    0.016029,
           0.015747,    0.015466,    0.015185,    0.014904,    0.014623,    0.014341,     0.01406,    0.013779,    0.013498,    0.013217,    0.012935,    0.012654,    0.012373,    0.012092,    0.011811,    0.011529,    0.011248,    0.010967,    0.010686,    0.010405,    0.010123,   0.0098422,    0.009561,
          0.0092798,   0.0089986,   0.0087174,   0.0084361,   0.0081549,   0.0078737,   0.0075925,   0.0073113,   0.0070301,   0.0067489,   0.0064677,   0.0061865,   0.0059053,   0.0056241,   0.0053429,   0.0050617,   0.0047805,   0.0044993,   0.0042181,   0.0039369,   0.0036557,   0.0033745,   0.0030933,
           0.002812,   0.0025308,   0.0022496,   0.0019684,   0.0016872,    0.001406,   0.0011248,  0.00084361,  0.00056241,   0.0002812,           0]]), 'Recall', 'Precision'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.11466,     0.11466,     0.15058,     0.17795,     0.20018,     0.21919,     0.23497,     0.24905,     0.26152,     0.27267,     0.28321,     0.29162,     0.30064,      0.3088,     0.31649,       0.323,     0.32933,     0.33584,      0.3417,     0.34704,     0.35247,     0.35725,     0.36245,
            0.36687,      0.3718,     0.37603,     0.38012,       0.384,     0.38802,     0.39183,     0.39514,     0.39928,     0.40207,     0.40575,     0.40927,     0.41243,     0.41527,     0.41822,     0.42153,     0.42422,      0.4271,     0.43011,     0.43248,     0.43537,     0.43797,     0.44034,
            0.44273,     0.44445,     0.44625,     0.44892,     0.45081,     0.45234,     0.45458,     0.45644,     0.45835,     0.46027,     0.46223,     0.46392,     0.46584,     0.46748,      0.4687,     0.47075,     0.47267,      0.4746,      0.4761,      0.4773,      0.4788,     0.48067,      0.4825,
            0.48379,     0.48496,     0.48612,     0.48712,     0.48831,     0.48965,     0.49084,     0.49232,     0.49343,     0.49424,     0.49531,     0.49705,     0.49815,     0.49892,     0.50021,     0.50114,     0.50255,      0.5035,     0.50436,     0.50549,     0.50643,     0.50728,     0.50789,
            0.50884,     0.50983,     0.51057,     0.51156,     0.51261,     0.51354,     0.51454,      0.5152,     0.51611,     0.51699,     0.51768,     0.51869,      0.5197,     0.52087,     0.52213,     0.52336,     0.52436,     0.52579,     0.52695,     0.52833,     0.52999,     0.53134,     0.53325,
            0.53434,     0.53617,     0.53748,     0.53963,     0.54098,     0.54269,      0.5444,     0.54575,     0.54757,     0.54866,      0.5505,     0.55158,     0.55311,     0.55518,     0.55606,      0.5568,     0.55789,     0.55935,     0.56029,     0.56142,     0.56305,     0.56501,     0.56681,
            0.56811,     0.56976,     0.57111,     0.57269,     0.57362,     0.57469,     0.57587,     0.57747,       0.578,     0.57916,     0.58008,     0.58119,     0.58219,     0.58346,     0.58497,     0.58628,     0.58758,     0.58858,     0.58956,     0.59063,     0.59115,      0.5918,     0.59314,
            0.59467,     0.59506,     0.59522,     0.59607,     0.59682,     0.59778,     0.59821,     0.59875,     0.59961,     0.60048,     0.60174,     0.60194,     0.60238,     0.60368,     0.60443,     0.60569,     0.60595,     0.60635,      0.6071,     0.60768,     0.60796,     0.60822,     0.60863,
            0.60872,     0.60909,     0.60977,     0.61049,     0.61098,     0.61171,     0.61172,     0.61201,     0.61296,     0.61346,     0.61355,     0.61465,     0.61518,     0.61555,     0.61624,     0.61673,     0.61759,     0.61797,     0.61854,      0.6191,     0.61909,     0.61919,     0.61949,
            0.62004,     0.62089,     0.62116,     0.62133,      0.6216,     0.62233,      0.6226,     0.62347,     0.62343,     0.62401,     0.62439,     0.62496,     0.62546,     0.62572,     0.62599,     0.62654,     0.62669,     0.62667,     0.62687,     0.62661,     0.62693,      0.6273,     0.62727,
             0.6276,     0.62785,     0.62793,     0.62832,     0.62848,      0.6286,     0.62895,     0.62912,     0.62913,     0.62929,      0.6294,     0.62916,     0.62892,     0.62908,     0.62927,     0.63014,     0.62952,     0.62947,     0.62953,     0.62953,     0.63003,     0.63012,     0.63046,
            0.63042,     0.63074,     0.63116,     0.63077,     0.63083,      0.6307,     0.63058,     0.63047,     0.63069,     0.63069,     0.63048,     0.63037,     0.63027,     0.63028,      0.6305,     0.63118,     0.63132,     0.63091,     0.63072,     0.63056,     0.63064,     0.63079,     0.63051,
            0.63017,     0.63006,     0.62994,     0.63041,      0.6305,     0.63058,     0.63078,     0.63065,     0.63068,      0.6303,     0.63026,     0.62994,     0.62983,     0.62986,     0.62992,     0.63002,     0.62975,     0.62981,     0.62929,     0.62896,     0.62902,     0.62854,      0.6286,
            0.62911,       0.629,      0.6292,     0.62896,     0.62876,     0.62869,     0.62841,      0.6283,     0.62815,     0.62817,     0.62807,     0.62784,     0.62771,      0.6275,     0.62749,     0.62766,     0.62749,     0.62733,     0.62726,     0.62743,     0.62714,     0.62697,       0.627,
            0.62696,     0.62667,     0.62636,     0.62597,     0.62567,     0.62549,     0.62546,     0.62522,     0.62499,     0.62475,     0.62468,     0.62449,     0.62451,     0.62428,     0.62403,     0.62375,     0.62397,     0.62381,     0.62363,     0.62368,     0.62387,     0.62331,     0.62323,
            0.62303,     0.62293,     0.62306,     0.62258,     0.62231,     0.62146,     0.62124,     0.62105,     0.62089,     0.62056,     0.62059,      0.6198,     0.61957,     0.61941,     0.61935,     0.61905,     0.61868,     0.61814,     0.61754,      0.6174,     0.61721,      0.6173,     0.61711,
            0.61714,     0.61699,     0.61651,     0.61642,     0.61615,      0.6158,     0.61532,     0.61505,     0.61491,     0.61429,     0.61399,     0.61353,     0.61327,     0.61313,     0.61282,     0.61279,      0.6122,     0.61203,     0.61177,     0.61129,     0.61127,       0.611,     0.61115,
            0.61114,     0.61078,     0.61072,     0.61026,     0.60999,     0.60985,     0.61017,     0.61002,     0.60976,     0.60931,     0.60909,     0.60908,     0.60837,     0.60802,     0.60779,     0.60723,     0.60631,     0.60601,     0.60585,     0.60544,     0.60504,     0.60484,     0.60419,
            0.60369,     0.60368,     0.60347,     0.60336,      0.6028,      0.6023,     0.60197,     0.60156,     0.60118,     0.60039,     0.60015,     0.60032,     0.59999,     0.59969,     0.59954,     0.59916,     0.59921,     0.59874,     0.59845,     0.59776,     0.59701,     0.59715,     0.59675,
            0.59626,     0.59601,     0.59551,     0.59505,     0.59498,     0.59407,      0.5929,     0.59247,     0.59223,     0.59165,     0.59115,     0.59057,     0.58941,      0.5885,     0.58803,     0.58752,     0.58732,     0.58638,     0.58608,      0.5858,     0.58548,     0.58472,     0.58417,
            0.58368,     0.58242,     0.58176,     0.58104,      0.5807,     0.58024,     0.57954,     0.57857,     0.57781,     0.57767,     0.57702,     0.57608,     0.57528,     0.57453,     0.57444,     0.57331,       0.573,     0.57247,     0.57202,     0.57156,     0.56973,     0.56892,     0.56789,
             0.5675,     0.56704,      0.5663,     0.56582,     0.56507,     0.56453,     0.56392,     0.56331,      0.5624,       0.562,     0.56133,     0.56037,     0.55966,       0.559,     0.55816,     0.55752,     0.55747,     0.55701,     0.55646,     0.55575,     0.55539,     0.55433,     0.55393,
            0.55315,     0.55214,     0.55173,     0.55114,      0.5507,     0.55027,     0.54923,     0.54865,     0.54756,     0.54687,      0.5465,     0.54594,     0.54528,     0.54424,     0.54336,     0.54212,     0.54202,     0.54096,     0.54058,     0.54026,      0.5389,     0.53851,     0.53732,
            0.53646,     0.53543,     0.53473,     0.53428,     0.53341,     0.53289,     0.53165,     0.53055,     0.52951,     0.52873,     0.52791,     0.52749,     0.52647,     0.52549,     0.52428,     0.52339,     0.52312,     0.52188,     0.52113,      0.5205,     0.51915,     0.51823,     0.51713,
            0.51535,     0.51472,     0.51382,     0.51313,     0.51209,     0.50986,     0.50926,     0.50835,     0.50706,     0.50608,     0.50495,       0.504,     0.50339,     0.50188,     0.50079,     0.49966,     0.49842,      0.4968,     0.49601,     0.49494,     0.49341,     0.49287,     0.49261,
             0.4919,     0.49118,     0.48996,     0.48873,     0.48818,     0.48559,     0.48496,      0.4838,     0.48345,      0.4826,     0.48101,      0.4803,     0.47961,     0.47887,     0.47822,     0.47765,     0.47672,     0.47571,     0.47521,      0.4742,     0.47249,     0.47018,     0.46947,
            0.46874,     0.46748,      0.4661,     0.46513,     0.46457,     0.46359,     0.46296,     0.46209,     0.46111,     0.45901,     0.45776,     0.45596,     0.45508,     0.45468,     0.45368,     0.45319,     0.45241,     0.45181,     0.45047,     0.44905,     0.44783,     0.44608,       0.444,
            0.44302,     0.44227,     0.44096,     0.43959,     0.43868,     0.43794,     0.43752,     0.43582,     0.43442,     0.43385,     0.43278,     0.43023,     0.42924,     0.42813,     0.42713,     0.42647,     0.42534,     0.42318,     0.42105,     0.41884,     0.41756,     0.41612,     0.41552,
            0.41431,     0.41312,     0.41236,     0.41032,      0.4086,     0.40637,     0.40596,     0.40525,     0.40351,     0.40255,     0.40138,     0.40007,     0.39817,     0.39698,       0.394,     0.39295,     0.39223,     0.39067,      0.3888,     0.38762,     0.38656,     0.38559,      0.3843,
             0.3831,     0.38177,      0.3807,     0.37808,     0.37761,     0.37653,     0.37529,     0.37467,       0.374,     0.37231,     0.36932,     0.36784,     0.36698,     0.36511,     0.36442,     0.36355,     0.36231,     0.36179,     0.36059,     0.35962,     0.35776,      0.3568,     0.35564,
            0.35472,     0.35334,     0.35207,     0.34898,     0.34692,     0.34522,     0.34395,     0.34287,     0.33976,     0.33927,     0.33746,     0.33607,     0.33595,     0.33331,     0.33281,     0.33041,     0.32846,     0.32817,     0.32638,     0.32559,     0.32359,     0.32149,     0.32105,
            0.31973,     0.31947,     0.31752,      0.3154,     0.31446,     0.31326,     0.31157,       0.311,     0.30858,     0.30729,     0.30656,     0.30441,     0.30206,      0.3019,     0.30037,     0.29794,      0.2978,     0.29546,      0.2945,     0.29257,     0.29132,     0.29089,     0.28927,
            0.28664,     0.28603,     0.28437,     0.28269,     0.28034,     0.27992,     0.27777,     0.27722,     0.27555,     0.27413,     0.27352,     0.27191,     0.27009,     0.26892,     0.26691,     0.26546,     0.26472,     0.26213,     0.26052,     0.25994,     0.25713,     0.25456,     0.25249,
            0.25227,     0.24999,     0.24777,      0.2447,     0.24367,     0.24195,     0.24033,     0.23955,      0.2379,      0.2367,     0.23501,     0.23455,     0.23353,     0.23215,     0.23095,     0.22836,     0.22714,     0.22697,     0.22552,     0.22435,     0.22243,      0.2202,     0.21968,
             0.2187,     0.21695,     0.21575,     0.21524,      0.2129,     0.21135,     0.21064,     0.20971,     0.20769,      0.2069,     0.20424,      0.2033,     0.20181,     0.19971,     0.19822,     0.19676,     0.19607,     0.19415,     0.19162,     0.18978,     0.18869,     0.18699,     0.18631,
            0.18441,      0.1824,     0.18128,     0.17928,      0.1783,     0.17725,     0.17631,     0.17427,     0.17176,     0.16914,     0.16844,     0.16671,     0.16442,     0.16244,     0.15944,     0.15738,     0.15547,     0.15322,     0.15081,      0.1501,     0.14823,     0.14613,     0.14486,
            0.14402,     0.14272,     0.14167,      0.1398,     0.13911,     0.13799,      0.1362,     0.13554,     0.13422,     0.13316,     0.13193,     0.13028,     0.12814,     0.12641,     0.12566,       0.124,     0.12218,     0.12153,     0.12101,     0.11918,     0.11701,     0.11601,     0.11416,
            0.11265,     0.11047,     0.10851,     0.10794,     0.10592,     0.10446,     0.10429,     0.10353,     0.10209,    0.099289,    0.098433,    0.097248,    0.095699,     0.09501,    0.094175,    0.093119,    0.092265,    0.090719,    0.089856,    0.088477,    0.086925,     0.08537,    0.083985,
           0.082317,    0.081554,    0.081014,     0.07929,    0.078411,    0.076501,    0.075624,    0.073703,    0.072824,    0.071769,    0.071244,     0.07065,    0.069655,    0.068947,    0.068364,     0.06801,    0.067345,    0.066109,     0.06484,    0.063166,    0.062568,    0.061502,    0.059905,
           0.058837,    0.057944,    0.055898,    0.054559,    0.053582,    0.052066,    0.051154,    0.049919,    0.048817,    0.047021,      0.0454,    0.043239,    0.041252,    0.040343,    0.039433,    0.038219,    0.036887,    0.034526,    0.033229,    0.031059,    0.028501,    0.028122,    0.026475,
           0.022805,    0.021512,    0.019849,    0.017446,    0.016145,    0.015517,    0.013354,    0.011862,    0.010554,   0.0096165,   0.0086777,   0.0079361,   0.0075424,   0.0066016,    0.006033,   0.0055384,   0.0050815,   0.0050023,   0.0049476,   0.0048929,   0.0045817,   0.0041009,   0.0037178,
          0.0033345,   0.0031261,   0.0030302,    0.002801,   0.0027462,   0.0026914,   0.0026366,   0.0023336,   0.0021741,   0.0020782,   0.0017438,   0.0013597,   0.0010128,  0.00089792,  0.00080184,  0.00028627,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0]]), 'Confidence', 'F1'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[   0.061877,    0.061877,    0.083404,     0.10061,     0.11514,     0.12798,       0.139,     0.14902,     0.15816,     0.16646,     0.17446,     0.18101,      0.1881,     0.19458,     0.20087,     0.20627,     0.21157,      0.2171,     0.22214,     0.22682,     0.23162,     0.23588,     0.24055,
            0.24461,     0.24913,     0.25303,     0.25688,      0.2606,     0.26441,     0.26804,     0.27135,     0.27536,     0.27822,     0.28181,     0.28533,     0.28856,     0.29152,     0.29448,     0.29796,     0.30076,     0.30373,     0.30692,      0.3095,     0.31263,     0.31541,     0.31802,
            0.32071,     0.32267,      0.3248,     0.32766,     0.32981,     0.33162,     0.33416,     0.33636,     0.33857,     0.34073,     0.34304,     0.34495,     0.34719,     0.34933,     0.35087,     0.35321,      0.3555,     0.35781,     0.35963,     0.36121,     0.36305,     0.36525,      0.3675,
            0.36915,     0.37068,     0.37219,      0.3735,     0.37509,     0.37683,     0.37844,     0.38028,     0.38178,     0.38304,     0.38447,     0.38665,     0.38811,     0.38916,     0.39091,     0.39219,     0.39407,      0.3954,     0.39668,     0.39813,     0.39937,     0.40067,     0.40168,
            0.40298,     0.40432,     0.40544,     0.40688,     0.40844,     0.40977,     0.41124,     0.41222,     0.41356,      0.4149,     0.41599,     0.41757,     0.41899,     0.42066,     0.42252,     0.42435,     0.42577,     0.42799,      0.4298,     0.43175,     0.43443,     0.43655,     0.43941,
             0.4412,     0.44391,       0.446,     0.44946,     0.45184,     0.45504,     0.45785,     0.46012,     0.46325,     0.46558,     0.46903,     0.47145,     0.47406,     0.47769,     0.47959,     0.48129,     0.48348,     0.48674,      0.4889,     0.49115,     0.49408,     0.49796,     0.50116,
            0.50375,      0.5067,     0.50936,     0.51238,     0.51459,     0.51722,     0.51962,     0.52315,     0.52489,     0.52726,      0.5293,     0.53192,     0.53457,     0.53725,     0.54008,     0.54308,       0.546,     0.54871,     0.55162,       0.554,     0.55548,     0.55794,     0.56105,
            0.56451,     0.56651,     0.56787,     0.57028,     0.57289,     0.57525,     0.57622,     0.57836,     0.58086,     0.58334,     0.58655,     0.58823,     0.58957,     0.59246,     0.59465,     0.59761,     0.59876,     0.60053,     0.60299,     0.60545,     0.60674,     0.60866,     0.61051,
            0.61145,     0.61346,      0.6163,     0.61837,     0.61996,     0.62228,     0.62384,     0.62503,     0.62752,      0.6297,     0.63067,      0.6339,     0.63578,     0.63707,     0.63956,     0.64104,     0.64402,     0.64584,     0.64743,     0.64967,     0.65045,     0.65232,     0.65351,
            0.65555,     0.65829,     0.65947,     0.66157,     0.66249,     0.66502,     0.66647,     0.66934,     0.67009,     0.67257,      0.6742,     0.67707,      0.6785,     0.68066,      0.6819,     0.68421,     0.68546,     0.68727,      0.6883,     0.68955,     0.69073,     0.69313,     0.69423,
            0.69636,     0.69732,     0.69844,     0.70068,     0.70206,      0.7038,     0.70496,     0.70671,     0.70761,     0.70887,     0.71069,     0.71124,     0.71335,     0.71463,     0.71572,     0.71896,       0.719,     0.71964,      0.7212,      0.7223,     0.72489,      0.7264,     0.72779,
            0.72949,     0.73048,     0.73176,     0.73301,     0.73375,     0.73429,     0.73581,     0.73649,     0.73846,     0.73931,     0.74015,      0.7408,     0.74276,     0.74339,     0.74444,     0.74703,     0.74813,     0.74866,      0.7498,     0.75061,     0.75171,      0.7527,     0.75371,
            0.75417,     0.75483,     0.75554,      0.7578,     0.75881,      0.7599,     0.76224,      0.7627,     0.76358,     0.76463,     0.76624,      0.7674,     0.76805,     0.76872,     0.77027,     0.77154,     0.77213,     0.77268,     0.77431,     0.77512,     0.77603,     0.77648,     0.77744,
            0.77965,     0.78033,     0.78136,     0.78208,     0.78356,     0.78507,     0.78564,      0.7863,     0.78709,     0.78909,     0.78964,     0.79031,     0.79055,     0.79176,     0.79277,     0.79392,     0.79454,     0.79514,     0.79557,     0.79656,     0.79742,     0.79775,     0.79885,
            0.79945,     0.79966,      0.8012,     0.80173,     0.80234,     0.80275,     0.80373,     0.80434,     0.80521,     0.80603,     0.80673,     0.80775,     0.80854,     0.80942,     0.80978,     0.81123,     0.81227,     0.81287,     0.81335,     0.81381,     0.81498,     0.81557,     0.81631,
            0.81704,     0.81766,     0.81905,     0.81972,     0.81993,     0.82111,     0.82138,     0.82223,     0.82267,     0.82383,     0.82484,       0.825,     0.82551,     0.82572,     0.82652,     0.82783,     0.82861,     0.82895,     0.82928,     0.83002,     0.83067,     0.83162,     0.83172,
            0.83294,     0.83349,      0.8348,     0.83602,     0.83652,     0.83689,     0.83711,     0.83752,     0.83793,     0.83815,     0.83929,     0.83952,     0.84082,     0.84145,     0.84198,     0.84257,     0.84281,     0.84348,     0.84455,     0.84479,     0.84645,     0.84689,     0.84876,
            0.84928,     0.84969,     0.85034,     0.85126,     0.85182,     0.85237,     0.85393,     0.85455,     0.85522,     0.85545,      0.8558,     0.85607,      0.8563,     0.85712,     0.85774,       0.858,     0.85872,     0.85915,      0.8598,     0.86037,     0.86067,     0.86108,     0.86131,
            0.86168,     0.86265,     0.86353,     0.86407,     0.86494,     0.86551,     0.86624,     0.86685,      0.8676,       0.869,     0.86968,     0.87074,     0.87107,     0.87186,     0.87226,     0.87264,     0.87365,      0.8743,     0.87493,     0.87474,      0.8747,     0.87585,     0.87622,
            0.87719,     0.87799,     0.87927,     0.87945,      0.8802,     0.88058,     0.88056,     0.88123,     0.88164,     0.88197,     0.88231,     0.88309,     0.88312,     0.88352,     0.88404,     0.88439,     0.88498,     0.88492,     0.88561,     0.88586,     0.88674,     0.88752,     0.88771,
            0.88856,     0.88826,     0.88875,     0.88858,     0.88898,      0.8892,     0.88952,     0.88978,     0.89026,     0.89122,      0.8914,     0.89184,       0.893,     0.89316,     0.89446,     0.89439,       0.895,     0.89538,     0.89528,     0.89603,     0.89648,     0.89681,     0.89692,
            0.89718,     0.89707,     0.89728,     0.89752,     0.89785,     0.89812,     0.89829,     0.89833,     0.89866,     0.89892,     0.90001,     0.90016,     0.90043,     0.90093,     0.90164,     0.90231,     0.90276,     0.90321,     0.90337,      0.9034,     0.90388,     0.90427,     0.90456,
            0.90476,     0.90565,     0.90594,     0.90619,     0.90628,     0.90694,     0.90672,     0.90713,     0.90712,     0.90735,     0.90784,      0.9083,     0.90873,     0.90909,     0.90987,      0.9102,     0.91018,     0.90996,     0.90988,      0.9112,     0.91149,     0.91141,      0.9114,
            0.91197,     0.91196,     0.91267,     0.91292,     0.91334,     0.91364,     0.91359,     0.91377,     0.91376,     0.91422,     0.91446,     0.91478,     0.91499,     0.91541,     0.91558,     0.91603,     0.91694,     0.91817,     0.91809,     0.91918,     0.91961,     0.92115,     0.92094,
            0.92081,     0.92113,      0.9216,     0.92232,     0.92257,      0.9227,     0.92326,     0.92409,     0.92408,     0.92412,     0.92391,     0.92418,     0.92474,     0.92469,     0.92472,     0.92519,     0.92496,     0.92489,     0.92497,     0.92524,     0.92518,     0.92574,     0.92597,
            0.92675,     0.92735,     0.92736,     0.92808,     0.92846,     0.92895,     0.92958,     0.92984,     0.93026,     0.93011,     0.93056,     0.93141,     0.93253,      0.9324,     0.93377,     0.93418,     0.93522,     0.93535,     0.93552,     0.93586,     0.93558,     0.93735,      0.9379,
            0.93804,     0.93809,     0.93813,     0.93837,     0.93867,     0.93851,     0.93946,     0.93932,     0.93916,     0.93909,     0.93943,     0.93947,     0.93953,     0.93955,     0.93958,     0.94004,     0.93992,     0.94037,     0.94043,     0.94047,     0.94028,     0.94027,     0.94022,
            0.94033,     0.94049,     0.94056,     0.94034,      0.9418,     0.94178,       0.942,     0.94173,     0.94208,     0.94199,      0.9424,     0.94317,     0.94331,     0.94343,     0.94356,     0.94376,     0.94418,     0.94414,     0.94441,     0.94497,     0.94477,     0.94516,     0.94507,
            0.94488,     0.94563,     0.94613,     0.94645,     0.94649,     0.94936,     0.94994,     0.94984,     0.94991,     0.95009,     0.95025,     0.95005,     0.94977,     0.94959,     0.94981,     0.94999,     0.94988,     0.94965,     0.95073,      0.9509,     0.95108,     0.95094,     0.95144,
            0.95196,     0.95176,     0.95196,     0.95273,     0.95328,     0.95348,     0.95366,     0.95389,     0.95456,     0.95469,     0.95463,     0.95533,     0.95615,     0.95664,      0.9573,     0.95718,     0.95739,      0.9577,     0.95831,     0.95856,     0.95947,     0.96013,     0.95998,
            0.95985,     0.95967,     0.95983,     0.96109,     0.96245,     0.96224,     0.96249,     0.96357,     0.96362,     0.96398,     0.96461,     0.96486,     0.96527,     0.96495,     0.96532,     0.96503,     0.96479,     0.96475,     0.96586,     0.96576,     0.96596,     0.96571,     0.96565,
            0.96594,     0.96591,      0.9675,     0.96818,     0.96807,     0.96793,     0.96843,     0.96908,     0.96976,     0.97057,     0.97049,     0.97025,     0.96999,     0.96997,     0.96979,     0.97101,       0.971,     0.97073,     0.97113,     0.97194,     0.97232,     0.97272,     0.97261,
            0.97338,     0.97331,     0.97314,     0.97403,      0.9754,     0.97536,     0.97515,     0.97509,     0.97658,     0.97645,     0.97639,     0.97623,     0.97662,      0.9765,     0.97688,     0.97674,     0.97666,     0.97699,     0.97683,     0.97677,     0.97709,     0.97683,     0.97662,
             0.9766,     0.97698,     0.97675,     0.97706,     0.97696,     0.97677,      0.9766,     0.97652,     0.97699,     0.97686,     0.97735,     0.97797,     0.97921,     0.97976,     0.98033,     0.98008,     0.98066,     0.98065,     0.98122,     0.98182,     0.98164,     0.98144,     0.98139,
             0.9813,     0.98157,     0.98251,     0.98246,     0.98225,     0.98211,     0.98204,     0.98272,     0.98488,     0.98481,     0.98619,     0.98612,     0.98601,     0.98831,     0.98905,     0.98896,     0.98892,      0.9888,     0.98864,     0.98852,     0.98844,     0.98833,     0.98828,
            0.98841,     0.98983,     0.98976,     0.99151,     0.99146,      0.9914,     0.99135,     0.99124,      0.9911,     0.99095,     0.99091,     0.99081,      0.9917,     0.99159,     0.99248,     0.99346,     0.99447,     0.99438,     0.99428,     0.99426,     0.99418,     0.99409,     0.99403,
            0.99399,     0.99394,     0.99455,     0.99503,     0.99501,     0.99496,     0.99489,     0.99487,     0.99481,     0.99607,     0.99603,     0.99598,     0.99591,     0.99585,     0.99582,     0.99576,     0.99569,     0.99567,      0.9971,     0.99705,     0.99699,     0.99696,     0.99691,
            0.99687,      0.9968,     0.99674,     0.99672,     0.99666,     0.99661,      0.9966,     0.99658,     0.99652,     0.99642,     0.99639,     0.99634,     0.99628,     0.99625,     0.99622,     0.99617,     0.99806,     0.99803,     0.99801,     0.99798,     0.99794,      0.9979,     0.99786,
            0.99782,      0.9978,     0.99778,     0.99773,     0.99771,     0.99765,     0.99762,     0.99755,     0.99752,     0.99749,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1,
                  1,           1,           1,           1,           1,           1,           1,           1,           1,           1,           1]]), 'Confidence', 'Precision'], [array([          0,    0.001001,    0.002002,    0.003003,    0.004004,    0.005005,    0.006006,    0.007007,    0.008008,    0.009009,     0.01001,    0.011011,    0.012012,    0.013013,    0.014014,    0.015015,    0.016016,    0.017017,    0.018018,    0.019019,     0.02002,    0.021021,    0.022022,    0.023023,
          0.024024,    0.025025,    0.026026,    0.027027,    0.028028,    0.029029,     0.03003,    0.031031,    0.032032,    0.033033,    0.034034,    0.035035,    0.036036,    0.037037,    0.038038,    0.039039,     0.04004,    0.041041,    0.042042,    0.043043,    0.044044,    0.045045,    0.046046,    0.047047,
          0.048048,    0.049049,     0.05005,    0.051051,    0.052052,    0.053053,    0.054054,    0.055055,    0.056056,    0.057057,    0.058058,    0.059059,     0.06006,    0.061061,    0.062062,    0.063063,    0.064064,    0.065065,    0.066066,    0.067067,    0.068068,    0.069069,     0.07007,    0.071071,
          0.072072,    0.073073,    0.074074,    0.075075,    0.076076,    0.077077,    0.078078,    0.079079,     0.08008,    0.081081,    0.082082,    0.083083,    0.084084,    0.085085,    0.086086,    0.087087,    0.088088,    0.089089,     0.09009,    0.091091,    0.092092,    0.093093,    0.094094,    0.095095,
          0.096096,    0.097097,    0.098098,    0.099099,      0.1001,      0.1011,      0.1021,      0.1031,      0.1041,     0.10511,     0.10611,     0.10711,     0.10811,     0.10911,     0.11011,     0.11111,     0.11211,     0.11311,     0.11411,     0.11512,     0.11612,     0.11712,     0.11812,     0.11912,
           0.12012,     0.12112,     0.12212,     0.12312,     0.12412,     0.12513,     0.12613,     0.12713,     0.12813,     0.12913,     0.13013,     0.13113,     0.13213,     0.13313,     0.13413,     0.13514,     0.13614,     0.13714,     0.13814,     0.13914,     0.14014,     0.14114,     0.14214,     0.14314,
           0.14414,     0.14515,     0.14615,     0.14715,     0.14815,     0.14915,     0.15015,     0.15115,     0.15215,     0.15315,     0.15415,     0.15516,     0.15616,     0.15716,     0.15816,     0.15916,     0.16016,     0.16116,     0.16216,     0.16316,     0.16416,     0.16517,     0.16617,     0.16717,
           0.16817,     0.16917,     0.17017,     0.17117,     0.17217,     0.17317,     0.17417,     0.17518,     0.17618,     0.17718,     0.17818,     0.17918,     0.18018,     0.18118,     0.18218,     0.18318,     0.18418,     0.18519,     0.18619,     0.18719,     0.18819,     0.18919,     0.19019,     0.19119,
           0.19219,     0.19319,     0.19419,      0.1952,      0.1962,      0.1972,      0.1982,      0.1992,      0.2002,      0.2012,      0.2022,      0.2032,      0.2042,     0.20521,     0.20621,     0.20721,     0.20821,     0.20921,     0.21021,     0.21121,     0.21221,     0.21321,     0.21421,     0.21522,
           0.21622,     0.21722,     0.21822,     0.21922,     0.22022,     0.22122,     0.22222,     0.22322,     0.22422,     0.22523,     0.22623,     0.22723,     0.22823,     0.22923,     0.23023,     0.23123,     0.23223,     0.23323,     0.23423,     0.23524,     0.23624,     0.23724,     0.23824,     0.23924,
           0.24024,     0.24124,     0.24224,     0.24324,     0.24424,     0.24525,     0.24625,     0.24725,     0.24825,     0.24925,     0.25025,     0.25125,     0.25225,     0.25325,     0.25425,     0.25526,     0.25626,     0.25726,     0.25826,     0.25926,     0.26026,     0.26126,     0.26226,     0.26326,
           0.26426,     0.26527,     0.26627,     0.26727,     0.26827,     0.26927,     0.27027,     0.27127,     0.27227,     0.27327,     0.27427,     0.27528,     0.27628,     0.27728,     0.27828,     0.27928,     0.28028,     0.28128,     0.28228,     0.28328,     0.28428,     0.28529,     0.28629,     0.28729,
           0.28829,     0.28929,     0.29029,     0.29129,     0.29229,     0.29329,     0.29429,      0.2953,      0.2963,      0.2973,      0.2983,      0.2993,      0.3003,      0.3013,      0.3023,      0.3033,      0.3043,     0.30531,     0.30631,     0.30731,     0.30831,     0.30931,     0.31031,     0.31131,
           0.31231,     0.31331,     0.31431,     0.31532,     0.31632,     0.31732,     0.31832,     0.31932,     0.32032,     0.32132,     0.32232,     0.32332,     0.32432,     0.32533,     0.32633,     0.32733,     0.32833,     0.32933,     0.33033,     0.33133,     0.33233,     0.33333,     0.33433,     0.33534,
           0.33634,     0.33734,     0.33834,     0.33934,     0.34034,     0.34134,     0.34234,     0.34334,     0.34434,     0.34535,     0.34635,     0.34735,     0.34835,     0.34935,     0.35035,     0.35135,     0.35235,     0.35335,     0.35435,     0.35536,     0.35636,     0.35736,     0.35836,     0.35936,
           0.36036,     0.36136,     0.36236,     0.36336,     0.36436,     0.36537,     0.36637,     0.36737,     0.36837,     0.36937,     0.37037,     0.37137,     0.37237,     0.37337,     0.37437,     0.37538,     0.37638,     0.37738,     0.37838,     0.37938,     0.38038,     0.38138,     0.38238,     0.38338,
           0.38438,     0.38539,     0.38639,     0.38739,     0.38839,     0.38939,     0.39039,     0.39139,     0.39239,     0.39339,     0.39439,      0.3954,      0.3964,      0.3974,      0.3984,      0.3994,      0.4004,      0.4014,      0.4024,      0.4034,      0.4044,     0.40541,     0.40641,     0.40741,
           0.40841,     0.40941,     0.41041,     0.41141,     0.41241,     0.41341,     0.41441,     0.41542,     0.41642,     0.41742,     0.41842,     0.41942,     0.42042,     0.42142,     0.42242,     0.42342,     0.42442,     0.42543,     0.42643,     0.42743,     0.42843,     0.42943,     0.43043,     0.43143,
           0.43243,     0.43343,     0.43443,     0.43544,     0.43644,     0.43744,     0.43844,     0.43944,     0.44044,     0.44144,     0.44244,     0.44344,     0.44444,     0.44545,     0.44645,     0.44745,     0.44845,     0.44945,     0.45045,     0.45145,     0.45245,     0.45345,     0.45445,     0.45546,
           0.45646,     0.45746,     0.45846,     0.45946,     0.46046,     0.46146,     0.46246,     0.46346,     0.46446,     0.46547,     0.46647,     0.46747,     0.46847,     0.46947,     0.47047,     0.47147,     0.47247,     0.47347,     0.47447,     0.47548,     0.47648,     0.47748,     0.47848,     0.47948,
           0.48048,     0.48148,     0.48248,     0.48348,     0.48448,     0.48549,     0.48649,     0.48749,     0.48849,     0.48949,     0.49049,     0.49149,     0.49249,     0.49349,     0.49449,      0.4955,      0.4965,      0.4975,      0.4985,      0.4995,      0.5005,      0.5015,      0.5025,      0.5035,
            0.5045,     0.50551,     0.50651,     0.50751,     0.50851,     0.50951,     0.51051,     0.51151,     0.51251,     0.51351,     0.51451,     0.51552,     0.51652,     0.51752,     0.51852,     0.51952,     0.52052,     0.52152,     0.52252,     0.52352,     0.52452,     0.52553,     0.52653,     0.52753,
           0.52853,     0.52953,     0.53053,     0.53153,     0.53253,     0.53353,     0.53453,     0.53554,     0.53654,     0.53754,     0.53854,     0.53954,     0.54054,     0.54154,     0.54254,     0.54354,     0.54454,     0.54555,     0.54655,     0.54755,     0.54855,     0.54955,     0.55055,     0.55155,
           0.55255,     0.55355,     0.55455,     0.55556,     0.55656,     0.55756,     0.55856,     0.55956,     0.56056,     0.56156,     0.56256,     0.56356,     0.56456,     0.56557,     0.56657,     0.56757,     0.56857,     0.56957,     0.57057,     0.57157,     0.57257,     0.57357,     0.57457,     0.57558,
           0.57658,     0.57758,     0.57858,     0.57958,     0.58058,     0.58158,     0.58258,     0.58358,     0.58458,     0.58559,     0.58659,     0.58759,     0.58859,     0.58959,     0.59059,     0.59159,     0.59259,     0.59359,     0.59459,      0.5956,      0.5966,      0.5976,      0.5986,      0.5996,
            0.6006,      0.6016,      0.6026,      0.6036,      0.6046,     0.60561,     0.60661,     0.60761,     0.60861,     0.60961,     0.61061,     0.61161,     0.61261,     0.61361,     0.61461,     0.61562,     0.61662,     0.61762,     0.61862,     0.61962,     0.62062,     0.62162,     0.62262,     0.62362,
           0.62462,     0.62563,     0.62663,     0.62763,     0.62863,     0.62963,     0.63063,     0.63163,     0.63263,     0.63363,     0.63463,     0.63564,     0.63664,     0.63764,     0.63864,     0.63964,     0.64064,     0.64164,     0.64264,     0.64364,     0.64464,     0.64565,     0.64665,     0.64765,
           0.64865,     0.64965,     0.65065,     0.65165,     0.65265,     0.65365,     0.65465,     0.65566,     0.65666,     0.65766,     0.65866,     0.65966,     0.66066,     0.66166,     0.66266,     0.66366,     0.66466,     0.66567,     0.66667,     0.66767,     0.66867,     0.66967,     0.67067,     0.67167,
           0.67267,     0.67367,     0.67467,     0.67568,     0.67668,     0.67768,     0.67868,     0.67968,     0.68068,     0.68168,     0.68268,     0.68368,     0.68468,     0.68569,     0.68669,     0.68769,     0.68869,     0.68969,     0.69069,     0.69169,     0.69269,     0.69369,     0.69469,      0.6957,
            0.6967,      0.6977,      0.6987,      0.6997,      0.7007,      0.7017,      0.7027,      0.7037,      0.7047,     0.70571,     0.70671,     0.70771,     0.70871,     0.70971,     0.71071,     0.71171,     0.71271,     0.71371,     0.71471,     0.71572,     0.71672,     0.71772,     0.71872,     0.71972,
           0.72072,     0.72172,     0.72272,     0.72372,     0.72472,     0.72573,     0.72673,     0.72773,     0.72873,     0.72973,     0.73073,     0.73173,     0.73273,     0.73373,     0.73473,     0.73574,     0.73674,     0.73774,     0.73874,     0.73974,     0.74074,     0.74174,     0.74274,     0.74374,
           0.74474,     0.74575,     0.74675,     0.74775,     0.74875,     0.74975,     0.75075,     0.75175,     0.75275,     0.75375,     0.75475,     0.75576,     0.75676,     0.75776,     0.75876,     0.75976,     0.76076,     0.76176,     0.76276,     0.76376,     0.76476,     0.76577,     0.76677,     0.76777,
           0.76877,     0.76977,     0.77077,     0.77177,     0.77277,     0.77377,     0.77477,     0.77578,     0.77678,     0.77778,     0.77878,     0.77978,     0.78078,     0.78178,     0.78278,     0.78378,     0.78478,     0.78579,     0.78679,     0.78779,     0.78879,     0.78979,     0.79079,     0.79179,
           0.79279,     0.79379,     0.79479,      0.7958,      0.7968,      0.7978,      0.7988,      0.7998,      0.8008,      0.8018,      0.8028,      0.8038,      0.8048,     0.80581,     0.80681,     0.80781,     0.80881,     0.80981,     0.81081,     0.81181,     0.81281,     0.81381,     0.81481,     0.81582,
           0.81682,     0.81782,     0.81882,     0.81982,     0.82082,     0.82182,     0.82282,     0.82382,     0.82482,     0.82583,     0.82683,     0.82783,     0.82883,     0.82983,     0.83083,     0.83183,     0.83283,     0.83383,     0.83483,     0.83584,     0.83684,     0.83784,     0.83884,     0.83984,
           0.84084,     0.84184,     0.84284,     0.84384,     0.84484,     0.84585,     0.84685,     0.84785,     0.84885,     0.84985,     0.85085,     0.85185,     0.85285,     0.85385,     0.85485,     0.85586,     0.85686,     0.85786,     0.85886,     0.85986,     0.86086,     0.86186,     0.86286,     0.86386,
           0.86486,     0.86587,     0.86687,     0.86787,     0.86887,     0.86987,     0.87087,     0.87187,     0.87287,     0.87387,     0.87487,     0.87588,     0.87688,     0.87788,     0.87888,     0.87988,     0.88088,     0.88188,     0.88288,     0.88388,     0.88488,     0.88589,     0.88689,     0.88789,
           0.88889,     0.88989,     0.89089,     0.89189,     0.89289,     0.89389,     0.89489,      0.8959,      0.8969,      0.8979,      0.8989,      0.8999,      0.9009,      0.9019,      0.9029,      0.9039,      0.9049,     0.90591,     0.90691,     0.90791,     0.90891,     0.90991,     0.91091,     0.91191,
           0.91291,     0.91391,     0.91491,     0.91592,     0.91692,     0.91792,     0.91892,     0.91992,     0.92092,     0.92192,     0.92292,     0.92392,     0.92492,     0.92593,     0.92693,     0.92793,     0.92893,     0.92993,     0.93093,     0.93193,     0.93293,     0.93393,     0.93493,     0.93594,
           0.93694,     0.93794,     0.93894,     0.93994,     0.94094,     0.94194,     0.94294,     0.94394,     0.94494,     0.94595,     0.94695,     0.94795,     0.94895,     0.94995,     0.95095,     0.95195,     0.95295,     0.95395,     0.95495,     0.95596,     0.95696,     0.95796,     0.95896,     0.95996,
           0.96096,     0.96196,     0.96296,     0.96396,     0.96496,     0.96597,     0.96697,     0.96797,     0.96897,     0.96997,     0.97097,     0.97197,     0.97297,     0.97397,     0.97497,     0.97598,     0.97698,     0.97798,     0.97898,     0.97998,     0.98098,     0.98198,     0.98298,     0.98398,
           0.98498,     0.98599,     0.98699,     0.98799,     0.98899,     0.98999,     0.99099,     0.99199,     0.99299,     0.99399,     0.99499,       0.996,       0.997,       0.998,       0.999,           1]), array([[    0.77974,     0.77974,     0.77383,     0.76979,     0.76567,     0.76285,     0.75919,     0.75769,     0.75488,     0.75328,     0.75188,     0.74972,     0.74841,     0.74765,     0.74568,       0.744,     0.74268,     0.74128,     0.73996,     0.73837,     0.73705,     0.73593,      0.7348,
            0.73349,     0.73246,     0.73171,     0.73068,     0.72936,     0.72871,     0.72805,     0.72664,     0.72598,     0.72467,      0.7243,     0.72355,     0.72265,     0.72158,     0.72129,     0.72026,     0.71961,     0.71923,     0.71848,     0.71764,     0.71679,     0.71632,     0.71557,
            0.71463,     0.71388,     0.71276,     0.71266,     0.71201,     0.71126,     0.71069,     0.70985,     0.70929,     0.70901,     0.70835,     0.70816,     0.70769,     0.70638,     0.70572,     0.70553,     0.70507,      0.7046,     0.70415,     0.70333,     0.70291,     0.70272,     0.70225,
            0.70169,     0.70113,     0.70056,     0.70009,     0.69944,     0.69887,     0.69822,     0.69794,     0.69737,     0.69644,     0.69597,     0.69568,     0.69531,     0.69493,     0.69437,      0.6939,     0.69343,     0.69296,     0.69231,     0.69212,     0.69193,     0.69118,     0.69043,
            0.69012,     0.68987,     0.68931,     0.68874,     0.68809,     0.68771,     0.68715,     0.68677,      0.6863,     0.68574,     0.68518,     0.68443,     0.68415,     0.68377,     0.68321,     0.68265,     0.68236,     0.68153,     0.68086,     0.68058,     0.67946,     0.67871,     0.67805,
            0.67733,     0.67683,     0.67617,     0.67505,     0.67392,     0.67214,     0.67129,     0.67054,     0.66942,     0.66782,     0.66623,     0.66453,     0.66379,     0.66266,     0.66154,     0.66041,     0.65938,     0.65741,      0.6561,     0.65516,     0.65441,     0.65291,     0.65225,
            0.65131,     0.65075,     0.64991,     0.64909,     0.64794,     0.64653,     0.64578,     0.64437,     0.64306,      0.6424,     0.64165,     0.64053,     0.63912,     0.63837,     0.63799,     0.63696,     0.63602,     0.63471,     0.63311,     0.63246,     0.63172,     0.63002,     0.62913,
            0.62824,     0.62664,     0.62533,      0.6243,     0.62284,     0.62214,     0.62195,     0.62064,     0.61961,     0.61867,     0.61773,     0.61631,     0.61576,     0.61533,     0.61454,     0.61398,     0.61332,     0.61229,     0.61126,     0.60992,     0.60919,     0.60779,     0.60675,
              0.606,     0.60478,     0.60338,     0.60281,     0.60225,      0.6015,     0.60006,     0.59953,     0.59906,     0.59803,     0.59733,     0.59653,     0.59587,     0.59543,     0.59456,     0.59421,     0.59325,      0.5924,     0.59212,     0.59128,     0.59062,     0.58927,     0.58884,
            0.58818,     0.58751,     0.58705,     0.58571,     0.58546,     0.58478,     0.58415,     0.58349,     0.58285,     0.58199,     0.58143,      0.5803,     0.58011,     0.57899,     0.57855,     0.57783,      0.5772,     0.57589,     0.57552,      0.5742,     0.57392,     0.57289,     0.57209,
             0.5712,     0.57097,     0.57036,     0.56951,     0.56886,     0.56792,     0.56773,     0.56689,     0.56632,     0.56578,      0.5648,     0.56407,     0.56237,     0.56182,     0.56144,     0.56085,     0.55985,     0.55938,     0.55854,     0.55788,     0.55713,     0.55638,      0.5561,
            0.55504,     0.55497,     0.55488,     0.55356,     0.55323,     0.55272,     0.55169,     0.55113,     0.55038,      0.5499,     0.54911,     0.54859,     0.54737,     0.54704,     0.54681,     0.54644,     0.54606,     0.54517,     0.54427,     0.54362,     0.54315,     0.54287,     0.54193,
            0.54118,     0.54068,     0.54015,     0.53968,     0.53931,     0.53888,     0.53799,     0.53758,     0.53719,     0.53612,     0.53527,     0.53424,     0.53377,      0.5335,     0.53283,     0.53236,     0.53171,     0.53154,     0.53002,     0.52917,     0.52884,     0.52795,     0.52759,
             0.5273,     0.52683,     0.52664,     0.52598,     0.52503,     0.52426,     0.52361,     0.52317,     0.52262,     0.52176,     0.52139,     0.52078,     0.52049,     0.51969,     0.51923,     0.51898,     0.51848,     0.51801,     0.51773,     0.51754,     0.51679,     0.51642,     0.51599,
            0.51569,     0.51521,     0.51415,     0.51341,     0.51276,     0.51235,     0.51191,     0.51135,     0.51069,     0.51004,     0.50966,     0.50901,     0.50872,     0.50807,      0.5076,     0.50666,     0.50654,      0.5061,     0.50567,     0.50556,     0.50537,     0.50441,     0.50401,
            0.50347,     0.50311,     0.50276,     0.50188,     0.50146,     0.49991,     0.49953,     0.49896,     0.49859,     0.49775,     0.49742,     0.49634,     0.49587,     0.49559,     0.49522,     0.49437,     0.49362,     0.49281,     0.49193,      0.4915,     0.49103,     0.49081,     0.49053,
            0.49016,     0.48978,     0.48872,     0.48818,     0.48768,     0.48711,     0.48643,     0.48596,     0.48565,     0.48481,     0.48405,      0.4834,     0.48264,     0.48227,     0.48171,     0.48149,     0.48068,     0.48025,     0.47958,     0.47892,     0.47836,     0.47789,     0.47749,
             0.4773,     0.47673,     0.47645,     0.47561,      0.4751,     0.47477,     0.47467,      0.4743,     0.47378,     0.47317,      0.4728,      0.4727,     0.47178,     0.47111,     0.47064,      0.4699,     0.46857,     0.46809,     0.46771,     0.46705,     0.46648,     0.46614,     0.46529,
            0.46459,     0.46429,     0.46379,     0.46351,      0.4626,     0.46184,     0.46126,      0.4606,     0.45994,     0.45862,     0.45816,     0.45807,     0.45758,     0.45702,     0.45673,     0.45619,     0.45598,     0.45525,     0.45476,       0.454,     0.45316,       0.453,     0.45244,
            0.45162,     0.45113,     0.45021,     0.44965,     0.44936,     0.44824,      0.4469,     0.44625,     0.44587,     0.44513,     0.44447,     0.44362,     0.44231,     0.44118,     0.44053,     0.43987,     0.43949,     0.43846,     0.43796,     0.43758,     0.43701,     0.43598,     0.43532,
            0.43457,     0.43325,     0.43241,     0.43165,     0.43118,     0.43062,     0.42977,     0.42864,      0.4277,     0.42732,     0.42657,     0.42544,     0.42431,     0.42347,     0.42308,     0.42187,      0.4214,     0.42073,     0.42027,     0.41961,     0.41754,      0.4166,     0.41547,
              0.415,     0.41453,      0.4137,     0.41313,     0.41227,     0.41163,     0.41095,     0.41029,     0.40926,     0.40879,     0.40785,     0.40681,       0.406,     0.40521,     0.40418,     0.40338,     0.40324,     0.40267,     0.40206,     0.40131,     0.40084,     0.39966,     0.39919,
            0.39834,     0.39712,     0.39665,     0.39599,     0.39552,     0.39495,     0.39392,     0.39325,     0.39213,     0.39138,     0.39091,     0.39025,      0.3895,     0.38837,     0.38733,     0.38602,     0.38592,     0.38489,     0.38451,     0.38396,     0.38254,     0.38216,     0.38096,
               0.38,     0.37896,     0.37814,     0.37764,      0.3767,     0.37614,     0.37492,     0.37379,     0.37275,     0.37191,     0.37106,     0.37059,     0.36956,     0.36852,      0.3673,     0.36636,     0.36595,     0.36454,     0.36382,     0.36304,     0.36166,     0.36053,     0.35949,
             0.3578,     0.35714,      0.3562,     0.35544,     0.35441,     0.35225,     0.35159,     0.35061,     0.34939,     0.34845,     0.34741,     0.34647,     0.34581,     0.34441,     0.34337,     0.34224,     0.34111,     0.33961,     0.33886,     0.33782,     0.33642,     0.33583,     0.33557,
             0.3348,     0.33406,     0.33293,     0.33171,     0.33115,     0.32871,     0.32805,     0.32696,     0.32659,     0.32583,     0.32433,     0.32358,     0.32282,     0.32217,     0.32141,     0.32085,     0.31989,     0.31897,      0.3185,     0.31755,     0.31605,     0.31379,     0.31309,
            0.31243,      0.3113,     0.31008,     0.30919,     0.30867,     0.30782,     0.30717,     0.30641,     0.30557,     0.30373,     0.30261,     0.30103,     0.30026,     0.29991,     0.29903,     0.29856,      0.2979,     0.29734,     0.29617,     0.29494,     0.29391,      0.2924,     0.29062,
            0.28977,     0.28911,     0.28798,     0.28685,     0.28593,      0.2853,     0.28493,     0.28352,     0.28229,     0.28182,     0.28088,     0.27868,     0.27783,     0.27689,     0.27604,     0.27548,      0.2745,      0.2727,     0.27092,     0.26904,     0.26801,     0.26679,     0.26631,
            0.26533,     0.26429,     0.26363,     0.26194,     0.26053,     0.25851,     0.25814,     0.25757,     0.25616,     0.25538,     0.25442,     0.25339,     0.25188,     0.25094,     0.24855,      0.2477,     0.24714,     0.24592,     0.24437,     0.24342,     0.24258,     0.24183,     0.24078,
             0.2398,     0.23877,     0.23792,     0.23583,     0.23543,     0.23459,     0.23361,     0.23311,     0.23256,     0.23125,     0.22895,     0.22777,     0.22706,     0.22561,     0.22504,     0.22438,     0.22343,     0.22302,     0.22208,     0.22133,     0.21987,     0.21912,     0.21825,
            0.21756,     0.21653,     0.21557,     0.21319,      0.2116,     0.21034,     0.20939,     0.20854,     0.20624,     0.20586,      0.2045,     0.20347,     0.20336,     0.20145,     0.20107,     0.19933,     0.19792,     0.19771,     0.19637,      0.1958,     0.19435,     0.19284,     0.19253,
            0.19157,     0.19138,     0.18993,     0.18838,     0.18772,     0.18687,     0.18565,     0.18522,     0.18349,     0.18254,     0.18203,     0.18052,     0.17888,     0.17877,      0.1777,     0.17596,     0.17587,     0.17425,     0.17357,      0.1722,     0.17132,     0.17101,      0.1699,
            0.16807,     0.16765,     0.16651,     0.16534,     0.16369,     0.16341,     0.16195,     0.16158,      0.1604,     0.15945,     0.15904,     0.15796,     0.15671,     0.15593,     0.15457,      0.1536,     0.15311,     0.15137,     0.15031,     0.14992,     0.14805,     0.14635,     0.14498,
            0.14484,     0.14334,     0.14188,     0.13986,     0.13919,     0.13808,     0.13703,     0.13652,     0.13544,     0.13467,     0.13356,     0.13325,     0.13257,     0.13168,     0.13089,     0.12923,     0.12844,     0.12834,      0.1274,     0.12664,     0.12542,     0.12401,     0.12368,
            0.12306,     0.12195,     0.12118,     0.12086,     0.11939,     0.11842,     0.11797,     0.11738,     0.11608,     0.11559,     0.11392,     0.11334,     0.11241,     0.11108,     0.11015,     0.10925,     0.10882,     0.10765,     0.10609,     0.10497,      0.1043,     0.10327,     0.10285,
            0.10169,     0.10045,    0.099776,    0.098552,     0.09796,    0.097323,    0.096761,    0.095531,    0.094025,    0.092461,    0.092046,    0.091009,    0.089639,    0.088467,     0.08668,    0.085458,    0.084325,    0.083007,    0.081596,     0.08118,    0.080085,    0.078861,    0.078124,
           0.077632,    0.076877,    0.076266,    0.075179,    0.074782,    0.074138,    0.073101,    0.072721,    0.071966,    0.071347,    0.070643,    0.069701,    0.068476,    0.067491,     0.06706,    0.066117,     0.06508,    0.064714,    0.064414,    0.063378,    0.062154,    0.061586,    0.060549,
           0.059701,    0.058476,    0.057378,     0.05706,    0.055929,     0.05512,    0.055024,    0.054602,      0.0538,    0.052247,    0.051774,    0.051119,    0.050263,    0.049883,    0.049423,    0.048842,    0.048368,    0.047519,    0.047046,     0.04629,    0.045442,    0.044593,    0.043837,
           0.042929,    0.042515,    0.042221,    0.041286,    0.040809,    0.039775,    0.039302,    0.038265,    0.037791,    0.037224,    0.036938,    0.036618,    0.036084,    0.035704,    0.035392,    0.035202,    0.034846,    0.034184,    0.033506,    0.032613,    0.032294,    0.031727,    0.030878,
            0.03031,    0.029836,    0.028753,    0.028045,    0.027528,    0.026729,    0.026248,    0.025598,    0.025019,    0.024076,    0.023227,    0.022097,     0.02106,    0.020587,    0.020113,    0.019482,     0.01879,    0.017566,    0.016895,    0.015774,    0.014456,    0.014261,    0.013415,
           0.011534,    0.010873,    0.010024,   0.0087996,   0.0081383,   0.0078191,   0.0067217,   0.0059666,   0.0053052,   0.0048315,   0.0043577,   0.0039838,   0.0037855,   0.0033118,   0.0030256,   0.0027769,   0.0025472,   0.0025074,   0.0024799,   0.0024525,   0.0022961,   0.0020547,   0.0018624,
          0.0016701,   0.0015655,   0.0015174,   0.0014025,    0.001375,   0.0013475,   0.0013201,   0.0011681,   0.0010882,   0.0010402,  0.00087264,  0.00068032,  0.00050664,  0.00044916,  0.00040108,  0.00014315,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0,
                  0,           0,           0,           0,           0,           0,           0,           0,           0,           0,           0]]), 'Confidence', 'Recall']]
fitness: 0.3838850285440719
keys: ['metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)', 'metrics/mAP50-95(B)']
maps: array([    0.35763])
names: {0: 'person'}
plot: True
results_dict: {'metrics/precision(B)': 0.7517119815271651, 'metrics/recall(B)': 0.5431519699812383, 'metrics/mAP50(B)': 0.6202204912054695, 'metrics/mAP50-95(B)': 0.3576255326928055, 'fitness': 0.3838850285440719}
save_dir: PosixPath('runs/detect/train3')
speed: {'preprocess': 0.1502678495949586, 'inference': 1.7710951639171417, 'loss': 0.0006012946554699784, 'postprocess': 1.0493961270250365}
task: 'detect'

# **Running inference on an image using the just-trained model**
# Run inference on an image with YOLOv8n
!yolo predict model=/kaggle/working/runs/detect/train3/weights/best.pt source='/kaggle/input/public-detection-dataset-for-yolov8/Test_img_2.jpg'
# **To Visulaize some testing images**
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the trained model
model = YOLO('/kaggle/working/runs/detect/train2/weights/best.pt')  # Replace with the path to your best.pt

# Specify the path to the image or directory of images you want to visualize
image_path = '/kaggle/input/public-detection-dataset-for-yolov8/Test_img_2.jpg'

# Perform inference and visualize results
results = model(image_path)

# Function to plot and display results
def display_results(result):
    # Plot the results
    plt.imshow(result.plot())
    plt.axis('off')
    plt.show()


# Check if results is a list and iterate over each result
if isinstance(results, list):
    for result in results:
        display_results(result)
else:
    display_results(results)
![image](https://github.com/user-attachments/assets/bd5e423e-5f7b-4745-b269-8bd43754fb43)

# **Changing the kernal directory to kaggle/working**
%cd /kaggle/working
# **Making The Zipp file of Outputs so that it can be saved in the local drive**
!zip -r output.zip /kaggle/working
